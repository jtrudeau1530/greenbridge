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
function fixture(count = 1) {
    const requests = [], added = [], disposed = [];
    let now = 0;
    class Group { add() {} }
    const context = vm.createContext({
        performance: { now: () => now },
        THREE: { MeshStandardMaterial: class {}, Group, DoubleSide: 2 },
        GLTFLoader: class { load(url, success, progress, failure) { requests.push({ url, success, failure }); } },
        scene: { add: obj => added.push(obj), remove() {} },
        camera: { position: { x: 0, y: 0, z: 0 } },
        world: { chunks: Array.from({ length: count }, (_, i) => ({
            id: `chunk_${i}`, url: `${i}.glb`, bounds: [i, 0, 0, i + 1, 1, 1],
        })) },
        buildTestChunk: () => ({ traverse() {} }),
    });
    vm.runInContext(source, context);
    added.length = 0; // Ignore the viewer's bounds-helper group.
    vm.runInContext('globalThis.state = { live, stream, drop, get inflight() { return inflight; } }', context);
    const object = () => ({ traverse(fn) { fn({ geometry: { dispose() { disposed.push(this); } } }); } });
    return { context, requests, added, disposed, state: context.state, object,
        frame(time = now) { now = time; context.state.stream(); },
        fail(index, time = now) { now = time; requests[index].failure(new Error('synthetic load failure')); },
    };
}

test('a transient failure retries only after backoff, then loads the chunk', () => {
    const f = fixture();
    f.frame();
    f.fail(0, 100);
    for (let i = 0; i < 600; i++) f.frame(1099);
    assert.equal(f.requests.length, 1);
    f.frame(1100);
    assert.equal(f.requests.length, 2);
    const obj = f.object();
    f.requests[1].success({ scene: obj });
    assert.equal(f.state.live.get('chunk_0').obj, obj);
    assert.equal(f.state.inflight, 0);
    assert.deepEqual(f.added, [obj]);
    f.frame(100000);
    assert.equal(f.requests.length, 2);
});

test('permanent failure stops after three attempts with increasing delays', () => {
    const f = fixture();
    f.frame(); f.fail(0);
    f.frame(1000); f.fail(1);
    f.frame(2999);
    assert.equal(f.requests.length, 2);
    f.frame(3000); f.fail(2);
    f.frame(1000000);
    assert.equal(f.requests.length, 3);
    assert.equal(f.state.inflight, 0);
});

test('leaving pauses retries; returning preserves the exhausted budget', () => {
    const f = fixture();
    f.frame(); f.fail(0);
    f.context.camera.position.x = 10000;
    f.frame(1000);
    assert.equal(f.requests.length, 1);
    f.context.camera.position.x = 0;
    f.frame(1000); f.fail(1);
    f.frame(3000); f.fail(2);
    f.context.camera.position.x = 10000; f.frame(100000);
    f.context.camera.position.x = 0; f.frame(100001);
    assert.equal(f.requests.length, 3);
});

test('pending requests are not duplicated and retries obey the inflight limit', () => {
    const f = fixture(6);
    f.frame();
    assert.equal(f.requests.length, 4);
    for (let i = 0; i < 600; i++) f.frame();
    assert.equal(f.requests.length, 4);
    f.fail(0); f.frame();
    assert.equal(f.requests.length, 5); // A fresh chunk can use the released slot during backoff.
    f.frame(1000);
    assert.equal(f.requests.length, 5);
    f.fail(1); f.frame();
    assert.equal(f.requests.length, 6);
    assert.equal(f.requests[5].url, '0.glb');
    assert.equal(f.state.inflight, 4);
});

test('successful chunks unload and can load again with a fresh attempt budget', () => {
    const f = fixture();
    f.frame(); f.fail(0); f.frame(1000);
    f.requests[1].success({ scene: f.object() });
    f.context.camera.position.x = 10000; f.frame();
    assert.equal(f.state.live.size, 0);
    assert.equal(f.disposed.length, 1);
    f.context.camera.position.x = 0; f.frame();
    assert.equal(f.requests.length, 3);
    assert.equal(f.state.live.get('chunk_0').attempts, 1);
});

test('stale success is disposed and stale failure cannot change a replacement entry', () => {
    const f = fixture();
    f.frame(); f.state.drop('chunk_0'); f.frame();
    const replacement = f.state.live.get('chunk_0');
    f.requests[0].success({ scene: f.object() });
    assert.equal(f.disposed.length, 1);
    assert.equal(f.state.live.get('chunk_0'), replacement);
    assert.equal(f.added.length, 0);
    f.state.drop('chunk_0'); f.frame();
    const latest = f.state.live.get('chunk_0');
    f.fail(1);
    assert.equal(f.state.live.get('chunk_0'), latest);
    assert.equal(latest.loading, true);
    assert.equal(latest.retryAt, 0);
    f.requests[2].success({ scene: f.object() });
    assert.equal(f.state.inflight, 0);
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
