const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Exercise the viewer's actual loading and streaming code without WebGL, a network or game data.
const html = readFileSync(join(__dirname, '..', 'viewer', 'index.html'), 'utf8');
const viewerModule = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1].replace(/^import .*;$/gm, '');
const worldBegin = html.indexOf('// ---------- world description ----------');
const worldEnd = html.indexOf('// Deterministic test terrain:', worldBegin);
assert(worldBegin >= 0 && worldEnd > worldBegin, 'viewer world description section must exist');
const worldSource = html.slice(worldBegin, worldEnd);

function loadWorld(fetch, search = '') {
    return vm.runInNewContext(worldSource + '\nloadWorld()', { fetch, location: { search }, URLSearchParams });
}

function manifestResponse(manifest) {
    return async () => ({ ok: true, json: async () => manifest });
}

const worldChunk = { id: 'cell', url: 'world/chunks/cell.glb', bounds: [-1, -2, -3, 1, 2, 3] };

test('a valid manifest is used, including an intentionally empty world', async () => {
    for (const chunks of [[worldChunk], []]) {
        const world = await loadWorld(manifestResponse({ spawn: [1, 2, 3], chunks }));
        assert.equal(world.test, false);
        assert.equal(world.spawn.join(','), '1,2,3');
        assert.deepEqual(world.chunks, chunks);
    }
});

test('malformed manifests keep the viewer usable and identify the invalid field', async t => {
    const cases = [
        ['missing spawn', {}, /spawn/],
        ['non-finite spawn', { spawn: [0, Infinity, 0], chunks: [] }, /spawn/],
        ['invalid chunks array', { spawn: [0, 0, 0], chunks: {} }, /chunks/],
        ['missing chunk', { spawn: [0, 0, 0], chunks: [null] }, /chunks\[0\]\.id/],
        ['missing URL', { spawn: [0, 0, 0], chunks: [{ ...worldChunk, url: '' }] }, /chunks\[0\]\.url/],
        ['bad bounds', { spawn: [0, 0, 0], chunks: [{ ...worldChunk, bounds: [0, 0, 0, -1, 1, 1] }] },
            /chunks\[0\]\.bounds/],
        ['duplicate chunk ID', { spawn: [0, 0, 0], chunks: [worldChunk, worldChunk] }, /chunks\[1\]\.id/],
        ['LOD tier without a band', { spawn: [0, 0, 0], chunks: [{ ...worldChunk, lod: 'hd', cell: [0, 0, 1, 1] }] },
            /chunks\[0\]\.lod/],
        ['LOD band out of order', { spawn: [0, 0, 0], lod_bands: { hd: [5, 1] },
            chunks: [{ ...worldChunk, lod: 'hd', cell: [0, 0, 1, 1] }] }, /chunks\[0\]\.lod/],
        ['LOD chunk without a cell', { spawn: [0, 0, 0], lod_bands: { hd: [0, 10] },
            chunks: [{ ...worldChunk, lod: 'hd' }] }, /chunks\[0\]\.cell/],
    ];
    for (const [label, manifest, field] of cases) {
        await t.test(label, async () => {
            const world = await loadWorld(manifestResponse(manifest));
            assert.equal(world.test, true);
            assert.ok(world.chunks.length > 0);
            assert.match(world.reason, field);
        });
    }
});

test('test mode, a missing manifest, and an unreadable manifest have distinct messages', async () => {
    const explicit = await loadWorld(async () => assert.fail('test mode must not fetch'), '?test');
    const missing = await loadWorld(async () => ({ ok: false, status: 404 }));
    const invalid = await loadWorld(async () => ({ ok: true, json: async () => { throw new SyntaxError(); } }));
    const unreachable = await loadWorld(async () => { throw new TypeError(); });
    for (const world of [explicit, missing, invalid, unreachable]) assert.equal(world.test, true);
    assert.match(explicit.reason, /requested/);
    assert.match(missing.reason, /not found/);
    assert.match(invalid.reason, /valid JSON/);
    assert.match(unreachable.reason, /could not be loaded/);
});

test('a WebGL startup error replaces the fly prompt and preserves the original error', async () => {
    const error = new Error('Error creating WebGL context.');
    const attributes = new Map();
    const start = { textContent: 'Click to fly', style: {}, setAttribute: (name, value) => attributes.set(name, value) };
    const startup = vm.runInNewContext(`(async () => { ${viewerModule} })()`, {
        THREE: { WebGLRenderer: class { constructor() { throw error; } } },
        document: { getElementById: id => { assert.equal(id, 'start'); return start; } },
    });
    await assert.rejects(startup, actual => actual === error);
    assert.match(start.textContent, /Could not start WebGL/);
    assert.equal(start.style.cursor, 'default');
    assert.equal(attributes.get('role'), 'alert');
});

test('the bounds overlay creates helpers on the first G press and reuses them', async () => {
    const chunks = [worldChunk, { id: 'other', url: 'other.glb', bounds: [5, 6, 7, 8, 9, 10] }];
    const handlers = new Map();
    const boundsGroup = await vm.runInNewContext(`(async () => { ${viewerModule}\nreturn boundsGroup; })()`, {
        THREE: {
            WebGLRenderer: class { domElement = { addEventListener() {} }; setPixelRatio() {} setSize() {} setAnimationLoop() {} },
            Scene: class { add() {} },
            PerspectiveCamera: class { position = { fromArray() {} }; },
            DirectionalLight: class { position = { set() {} }; },
            Color: class {}, Fog: class {}, HemisphereLight: class {}, Clock: class {}, MeshStandardMaterial: class {},
            Vector3: class { constructor(...xyz) { this.xyz = xyz; } },
            Box3: class { constructor(min, max) { Object.assign(this, { min, max }); } },
            Box3Helper: class { constructor(box, color) { Object.assign(this, { box, color }); } },
            Group: class { children = []; add(child) { this.children.push(child); } },
        },
        document: { body: { prepend() {} }, getElementById: () => ({}), addEventListener() {} },
        addEventListener: (type, handler) => handlers.set(type, handler),
        devicePixelRatio: 1, innerWidth: 800, innerHeight: 600, URLSearchParams, location: { search: '' },
        fetch: manifestResponse({ spawn: [0, 0, 0], chunks }), GLTFLoader: class {},
    });
    const pressG = (repeat = false) => handlers.get('keydown')({ code: 'KeyG', repeat });
    assert.equal(boundsGroup.visible, false);
    assert.equal(boundsGroup.children.length, 0);
    pressG();
    assert.equal(boundsGroup.visible, true);
    const helpers = [...boundsGroup.children];
    assert.deepEqual(helpers.map(({ box }) => [...box.min.xyz, ...box.max.xyz]), chunks.map(c => c.bounds));
    assert.ok(helpers.every(helper => helper.color === 0x5fb3ff));
    pressG(true);
    assert.equal(boundsGroup.visible, true);
    pressG();
    assert.equal(boundsGroup.visible, false);
    pressG();
    assert.equal(boundsGroup.visible, true);
    assert.equal(boundsGroup.children.length, helpers.length);
    helpers.forEach((helper, i) => assert.equal(boundsGroup.children[i], helper));
});

const begin = html.indexOf('// ---------- chunk streaming ----------');
const end = html.indexOf('// ---------- main loop ----------', begin);
assert(begin >= 0 && end > begin, 'viewer streaming section must exist');
const source = html.slice(begin, end);
const settle = () => new Promise(setImmediate);
function fixture(count = 1) {
    const requests = [], added = [], disposed = [];
    let now = 0;
    class Group { add() {} }
    const context = vm.createContext({
        performance: { now: () => now },
        THREE: { MeshStandardMaterial: class {}, Group, DoubleSide: 2,
            LoaderUtils: { extractUrlBase: url => url.slice(0, url.lastIndexOf('/') + 1) || './' } },
        AbortController,
        fetch: (url, { signal }) => new Promise((resolve, reject) => {
            const request = { url, signal, reject };
            request.headers = (url = '') => resolve({ ok: true, url, arrayBuffer: () => new Promise((body, fail) => {
                request.completeBody = () => body(request);
                signal.addEventListener('abort', () => fail(signal.reason), { once: true });
            }) });
            signal.addEventListener('abort', () => reject(signal.reason), { once: true });
            requests.push(request);
        }),
        GLTFLoader: class {
            parseAsync(request, resourcePath) {
                return new Promise((resolve, reject) => { request.parse = { resolve, reject, resourcePath }; });
            }
        },
        scene: { add: obj => added.push(obj), remove() {} },
        camera: { position: { x: 0, y: 0, z: 0 } },
        world: { test: false, chunks: Array.from({ length: count }, (_, i) => ({
            id: `chunk_${i}`, url: `${i}.glb`, bounds: [i, 0, 0, i + 1, 1, 1],
        })) },
        buildTestChunk: ([x, z, size]) => ({ x, z, size, traverse() {} }),
    });
    vm.runInContext(source, context);
    added.length = 0; // Ignore the viewer's bounds-helper group.
    vm.runInContext(`globalThis.state = { live, stream, drop,
        get inflight() { return inflight; }, get failed() { return failed; } }`, context);
    const object = () => ({ traverse(fn) { fn({ geometry: { dispose() { disposed.push(this); } } }); } });
    return { context, requests, added, disposed, state: context.state, object,
        frame(time = now) { now = time; context.state.stream(); },
        async fail(index, time = now) {
            now = time; requests[index].reject(new Error('synthetic load failure')); await settle();
        },
        async receive(index, responseUrl) {
            requests[index].headers(responseUrl); await settle(); requests[index].completeBody(); await settle();
        },
        async finish(index, obj) { requests[index].parse.resolve({ scene: obj }); await settle(); },
        async succeed(index, obj) { await this.receive(index); await this.finish(index, obj); },
    };
}

test('chunk dependencies resolve relative to the final response directory', async t => {
    const cases = [
        ['redirected chunk', 'world/source/cell.glb',
            'https://cdn.example/served/cell.glb?signature=one/two', 'https://cdn.example/served/'],
        ['response without a URL', 'world/source/cell.glb?signature=one/two', '', 'world/source/'],
    ];
    for (const [label, url, responseUrl, resourcePath] of cases) {
        await t.test(label, async () => {
            const f = fixture();
            f.context.world.chunks[0].url = url;
            f.frame();
            assert.equal(f.requests[0].url, url);
            await f.receive(0, responseUrl);
            assert.equal(f.requests[0].parse.resourcePath, resourcePath);
            const obj = f.object();
            await f.finish(0, obj);
            assert.equal(f.state.live.get('chunk_0').obj, obj);
        });
    }
});

test('a transient failure retries only after backoff, then loads the chunk', async () => {
    const f = fixture();
    f.frame();
    await f.fail(0, 100);
    for (let i = 0; i < 600; i++) f.frame(1099);
    assert.equal(f.requests.length, 1);
    f.frame(1100);
    assert.equal(f.requests.length, 2);
    const obj = f.object();
    await f.succeed(1, obj);
    assert.equal(f.state.live.get('chunk_0').obj, obj);
    assert.equal(f.state.inflight, 0);
    assert.deepEqual(f.added, [obj]);
    f.frame(100000);
    assert.equal(f.requests.length, 2);
});

test('permanent failure stops after three attempts with increasing delays', async () => {
    const f = fixture();
    f.frame(); await f.fail(0);
    f.frame(1000); await f.fail(1);
    f.frame(2999);
    assert.equal(f.requests.length, 2);
    f.frame(3000); await f.fail(2);
    f.frame(1000000);
    assert.equal(f.requests.length, 3);
    assert.equal(f.state.inflight, 0);
});

test('leaving pauses retries; returning preserves the exhausted budget', async () => {
    const f = fixture();
    f.frame(); await f.fail(0);
    f.context.camera.position.x = 10000;
    f.frame(1000);
    assert.equal(f.requests.length, 1);
    f.context.camera.position.x = 0;
    f.frame(1000); await f.fail(1);
    f.frame(3000); await f.fail(2);
    f.context.camera.position.x = 10000; f.frame(100000);
    f.context.camera.position.x = 0; f.frame(100001);
    assert.equal(f.requests.length, 3);
});

test('a full load queue skips looking up unloaded chunk bounds until a slot opens', async () => {
    const f = fixture(8);
    f.frame();
    assert.equal(f.state.inflight, 4);
    const next = f.context.world.chunks[4];
    const bounds = next.bounds;
    let reads = 0;
    Object.defineProperty(next, 'bounds', { get() { reads++; return bounds; } });

    f.frame();
    assert.equal(reads, 0);
    assert.equal(f.requests.length, 4);

    await f.succeed(0, f.object());
    f.frame();
    assert.equal(reads, 1);
    assert.equal(f.requests[4].url, '4.glb');
});

test('candidate distances are computed once and nearest chunks load first', () => {
    const f = fixture(7);
    const distances = [100, 5, 20, 20, 10, 1400, 4];
    const reads = Array(distances.length).fill(0);
    f.context.world.chunks.forEach((chunk, i) => {
        const x = distances[i];
        const bounds = [x, 0, 0, x + 1, 1, 1];
        Object.defineProperty(chunk, 'bounds', { get() { reads[i]++; return bounds; } });
    });

    f.frame();
    assert.deepEqual(f.requests.map(r => r.url), ['6.glb', '1.glb', '4.glb', '2.glb']);
    assert.deepEqual(reads, Array(distances.length).fill(1));
});

test('pending requests are not duplicated and retries obey the inflight limit', async () => {
    const f = fixture(6);
    f.frame();
    assert.equal(f.requests.length, 4);
    for (let i = 0; i < 600; i++) f.frame();
    assert.equal(f.requests.length, 4);
    await f.fail(0); f.frame();
    assert.equal(f.requests.length, 5); // A fresh chunk can use the released slot during backoff.
    f.frame(1000);
    assert.equal(f.requests.length, 5);
    await f.fail(1); f.frame();
    assert.equal(f.requests.length, 6);
    assert.equal(f.requests[5].url, '0.glb');
    assert.equal(f.state.inflight, 4);
});

test('successful chunks unload and can load again with a fresh attempt budget', async () => {
    const f = fixture();
    f.frame(); await f.fail(0); f.frame(1000);
    await f.succeed(1, f.object());
    f.context.camera.position.x = 10000; f.frame();
    assert.equal(f.state.live.size, 0);
    assert.equal(f.disposed.length, 1);
    f.context.camera.position.x = 0; f.frame();
    assert.equal(f.requests.length, 3);
    assert.equal(f.state.live.get('chunk_0').attempts, 1);
});

test('clay, wireframe and eviction preserve imported material metadata', async t => {
    const f = fixture();
    const materials = [{ dispose: t.mock.fn() }, { dispose: t.mock.fn() }];
    const userData = { origMaterial: { label: 'imported glTF extras' } };
    const mesh = { isMesh: true, material: materials, userData };

    f.frame();
    await f.succeed(0, { traverse: visit => visit(mesh) });
    assert.equal(mesh.material, materials);

    vm.runInContext('clay = true; applyClay(); wire = true; applyWire()', f.context);
    assert.notEqual(mesh.material, materials);
    assert.equal(mesh.material.wireframe, true);
    for (const material of materials) assert.equal(material.wireframe, true);

    vm.runInContext('clay = false; applyClay()', f.context);
    assert.equal(mesh.material, materials);
    vm.runInContext('clay = true; applyClay()', f.context);
    f.state.drop('chunk_0');
    for (const material of materials) assert.equal(material.dispose.mock.callCount(), 1);
    assert.deepEqual(userData, { origMaterial: { label: 'imported glTF extras' } });
});

test('stale and evicted chunks close each shared bitmap once', async t => {
    const f = fixture();
    class Bitmap { close = t.mock.fn(); }
    f.context.ImageBitmap = Bitmap;
    vm.runInContext('clay = true', f.context);
    const otherClose = t.mock.fn();
    function chunk() {
        const bitmap = new Bitmap();
        const texture = image => ({ isTexture: true, image, dispose: t.mock.fn() });
        const textures = [texture(bitmap), texture(bitmap), texture({ close: otherClose })];
        const material = { map: textures[0], normalMap: textures[1], alphaMap: textures[2], dispose() {} };
        const mesh = { isMesh: true, material, userData: {} };
        return { bitmap, textures, traverse: visit => visit(mesh) };
    }

    const stale = chunk(), live = chunk();
    f.frame(); await f.receive(0);
    f.state.drop('chunk_0'); f.frame();
    await f.succeed(1, live);
    await f.finish(0, stale);
    assert.equal(stale.bitmap.close.mock.callCount(), 1);
    assert.equal(live.bitmap.close.mock.callCount(), 0);
    assert.equal(f.state.live.get('chunk_0').obj, live);

    f.context.camera.position.x = 10000; f.frame();
    assert.equal(live.bitmap.close.mock.callCount(), 1);
    assert.equal(otherClose.mock.callCount(), 0);
    for (const tex of [...stale.textures, ...live.textures]) assert.equal(tex.dispose.mock.callCount(), 1);
});

test('stale parses retain their slot, then dispose success or ignore failure', async () => {
    const f = fixture();
    f.frame(); await f.receive(0); f.state.drop('chunk_0'); f.frame();
    const replacement = f.state.live.get('chunk_0');
    assert.equal(f.state.inflight, 2);
    await f.finish(0, f.object());
    assert.equal(f.disposed.length, 1);
    assert.equal(f.state.live.get('chunk_0'), replacement);
    assert.equal(f.added.length, 0);
    await f.receive(1); f.state.drop('chunk_0'); f.frame();
    const latest = f.state.live.get('chunk_0');
    f.requests[1].parse.reject(new Error('synthetic parse failure'));
    await settle();
    assert.equal(f.state.live.get('chunk_0'), latest);
    assert.equal(latest.loading, true);
    assert.equal(latest.retryAt, 0);
    assert.equal(f.state.failed, 0);
    await f.succeed(2, f.object());
    assert.equal(f.state.inflight, 0);
});

test('moving far away cancels old downloads so the new area can load', async () => {
    const f = fixture(5);
    f.context.world.chunks[4].bounds = [10000, 0, 0, 10001, 1, 1];
    f.frame();
    assert.equal(f.requests.length, 4);
    f.requests[0].headers(); // This request is already reading the GLB body.
    await settle();
    assert(f.requests[0].completeBody);
    f.context.camera.position.x = 10000;
    f.frame();
    assert(f.requests.slice(0, 4).every(request => request.signal.aborted));
    await settle();
    f.frame();
    assert.equal(f.requests.length, 5);
    assert.equal(f.requests[4].url, '4.glb');
    assert.equal(f.requests[0].parse, undefined);
    assert.equal(f.state.inflight, 1);
    assert.equal(f.state.failed, 0);
});

test('canceling a retry preserves the failures already counted', async () => {
    const f = fixture();
    f.frame(); await f.fail(0);
    f.frame(1000);
    f.context.camera.position.x = 10000; f.frame(); await settle();
    assert.equal(f.state.failed, 1);
    f.context.camera.position.x = 0; f.frame();
    assert.equal(f.requests.length, 3);
    await f.fail(2);
    f.frame(2999);
    assert.equal(f.requests.length, 3);
    f.frame(3000); await f.fail(3);
    f.frame(100000);
    assert.equal(f.requests.length, 4);
    assert.equal(f.state.failed, 3);
});

test('manifest chunks with test metadata still load their GLB', async t => {
    const cases = [['truthy flag', true], ['terrain-like tuple', [0, 0, 256]]];
    for (const [label, value] of cases) {
        await t.test(label, async () => {
            const f = fixture();
            const chunk = { ...worldChunk, test: value };
            f.context.world = await loadWorld(manifestResponse({ spawn: [0, 0, 0], chunks: [chunk] }));
            f.frame();
            assert.deepEqual(f.requests.map(request => request.url), [chunk.url]);
            const object = f.object();
            await f.succeed(0, object);
            assert.equal(f.state.live.get(chunk.id).obj, object);
            assert.deepEqual(f.added, [object]);
        });
    }
});

test('generated test terrain loads without network requests', () => {
    const f = fixture();
    f.context.world.test = true;
    f.context.world.chunks[0].test = [0, 0, 256];
    f.frame(); f.frame();
    assert.equal(f.requests.length, 0);
    assert.equal(f.state.inflight, 0);
    assert(f.state.live.get('chunk_0').obj);
    assert.equal(f.state.live.get('chunk_0').loading, false);
});

test('a valid LOD manifest is accepted', async () => {
    const manifest = { spawn: [0, 0, 0], lod_bands: { hd: [0, 130], lod: [130, 450] },
        chunks: [{ ...worldChunk, lod: 'hd', cell: [0, 0, 256, 256] }] };
    const world = await loadWorld(manifestResponse(manifest));
    assert.equal(world.test, false);
});

test('LOD sets show one tier per cell by horizontal distance, with slack before unloading', async () => {
    const f = fixture(0);
    const cell = [0, 0, 256, 256];
    f.context.world = { test: false, lod_bands: { hd: [0, 130], lod: [130, 450], slod1: [450, 1000] }, chunks: [
        { id: 'hd', url: 'hd.glb', bounds: [0, 0, 0, 256, 50, 256], lod: 'hd', cell },
        { id: 'lod', url: 'lod.glb', bounds: [0, 0, 0, 256, 80, 256], lod: 'lod', cell },
        { id: 'slod1', url: 'slod1.glb', bounds: [0, 0, 0, 256, 90, 256], lod: 'slod1', cell },
    ] };
    const pos = f.context.camera.position;
    Object.assign(pos, { x: 300, y: 500, z: 100 });          // 44 m east of the cell, high up: still HD
    f.frame();
    assert.deepEqual(f.requests.map(r => r.url), ['hd.glb']);
    const hd = Object.assign(f.object(), { visible: true });
    const lod = Object.assign(f.object(), { visible: true });
    await f.succeed(0, hd);
    Object.assign(pos, { x: 256 + 150, y: 0, z: 100 });      // 150 m: LOD replaces HD (HD kept inside the slack)
    f.frame();
    assert.deepEqual(f.requests.map(r => r.url), ['hd.glb', 'lod.glb']);
    assert.ok(f.state.live.has('hd'));
    await f.succeed(1, lod);
    f.frame();
    assert.equal(hd.visible, false);
    assert.equal(lod.visible, true);
    pos.x = 256 + 120;                                    // back inside HD, with LOD retained
    f.frame();
    assert.equal(hd.visible, true);
    assert.equal(lod.visible, false);
    assert.equal(f.requests.length, 2);
    Object.assign(pos, { x: 256 + 200, y: 0, z: 100 });      // past the slack: HD unloads
    f.frame();
    assert.ok(!f.state.live.has('hd'));
    assert.ok(f.state.live.has('lod'));
    Object.assign(pos, { x: 256 + 600, y: 0, z: 100 });      // 600 m: SLOD1 only
    f.frame();
    assert.ok(!f.state.live.has('lod'));
    assert.equal(f.requests.at(-1).url, 'slod1.glb');
});

