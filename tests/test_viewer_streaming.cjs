const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Exercise the viewer's actual streaming code without WebGL, a network or game data.
const html = readFileSync(join(__dirname, '..', 'viewer', 'index.html'), 'utf8');
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
            request.headers = () => resolve({ ok: true, arrayBuffer: () => new Promise((body, fail) => {
                request.completeBody = () => body(request);
                signal.addEventListener('abort', () => fail(signal.reason), { once: true });
            }) });
            signal.addEventListener('abort', () => reject(signal.reason), { once: true });
            requests.push(request);
        }),
        GLTFLoader: class {
            parseAsync(request) {
                return new Promise((resolve, reject) => { request.parse = { resolve, reject }; });
            }
        },
        scene: { add: obj => added.push(obj), remove() {} },
        camera: { position: { x: 0, y: 0, z: 0 } },
        world: { chunks: Array.from({ length: count }, (_, i) => ({
            id: `chunk_${i}`, url: `${i}.glb`, bounds: [i, 0, 0, i + 1, 1, 1],
        })) },
        buildTestChunk: () => ({ traverse() {} }),
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
        async receive(index) {
            requests[index].headers(); await settle(); requests[index].completeBody(); await settle();
        },
        async finish(index, obj) { requests[index].parse.resolve({ scene: obj }); await settle(); },
        async succeed(index, obj) { await this.receive(index); await this.finish(index, obj); },
    };
}

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

test('generated test terrain loads without network requests', () => {
    const f = fixture();
    f.context.world.chunks[0].test = [0, 0, 256];
    f.frame(); f.frame();
    assert.equal(f.requests.length, 0);
    assert.equal(f.state.inflight, 0);
    assert(f.state.live.get('chunk_0').obj);
    assert.equal(f.state.live.get('chunk_0').loading, false);
});
