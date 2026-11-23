const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = readFileSync(join(__dirname, '..', 'viewer', 'studio.html'), 'utf8');
const script = html.match(/<script>\n([\s\S]*?)<\/script>/)[1];

function fakeElement() {
    return {
        children: [], style: {}, innerHTML: '', innerText: '', value: '',
        classList: { add() {}, remove() {}, toggle() {} },
        appendChild(child) { this.children.push(child); },
        replaceChildren() { this.children = []; },
        get childElementCount() { return this.children.length; },
        remove() {},
    };
}

class Vector3 { set() { return this; } sub() { return this; } }
class Disposable { disposed = false; dispose() { this.disposed = true; } }

function loadStudio() {
    const elements = new Map(), loads = [], alerts = [];
    const document = {
        getElementById: id => elements.get(id) ?? elements.set(id, fakeElement()).get(id),
        createElement: fakeElement,
        querySelectorAll: () => [],
    };
    const context = vm.createContext({
        document, URLSearchParams, localStorage: { getItem: () => null }, window: {}, setTimeout() {},
        alert: message => alerts.push(message), console: { error() {} },
        THREE: {
            MeshStandardMaterial: Disposable, MeshNormalMaterial: Disposable, Vector3,
            Box3: class { setFromObject() { return this; } getCenter(v) { return v; } getSize(v) { return v; } },
            GLTFLoader: class { load(url, onLoad, onProgress, onError) { loads.push({ url, onLoad, onError }); } },
        },
    });
    vm.runInContext(script, context);
    vm.runInContext(`
        scene = { objects: new Set(), add(o) { this.objects.add(o); }, remove(o) { this.objects.delete(o); } };
        camera = { position: new THREE.Vector3() };
        controls = { target: new THREE.Vector3(), update() {} };`, context);
    return { context, elements, loads, alerts, run: code => vm.runInContext(code, context) };
}

function fakeModel() {
    const geometry = Object.assign(new Disposable(), { attributes: { position: { count: 3 } }, index: null });
    const texture = Object.assign(new Disposable(), { isTexture: true });
    const material = Object.assign(new Disposable(), { map: texture });
    const mesh = { isMesh: true, geometry, material, userData: {} };
    return { position: new Vector3(), userData: {}, parts: [geometry, material, texture],
        traverse(visit) { visit(this); visit(mesh); } };
}

const HOSTILE = '"><img src=x onerror=alert(1)>';
const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" };
const tagShape = markup => markup.match(/<[^>]*>/g).map(tag => tag.replace(/"[^"]*"/g, '""'));
const visibleText = markup => markup.replace(/<[^>]*>/g, '').replace(/&(?:#(\d+)|#x([\da-f]+)|([a-z]+));/gi,
    (_, dec, hex, name) => dec ? String.fromCharCode(dec) : hex ? String.fromCharCode(parseInt(hex, 16)) : ENTITIES[name]);

function renderModelList(name) {
    const studio = loadStudio();
    studio.context.models = [{ name: name + '.glb', folder: name, category: name, size: 4, url: '/models/x.glb' }];
    studio.run('allModels = models; renderAccordionList();');
    const [group] = studio.elements.get('model-list').children;
    return group.children[0].innerHTML + group.children[1].children[0].innerHTML;
}

function renderTextureGrid(name) {
    const studio = loadStudio();
    studio.context.textures = Array.from({ length: 81 },
        () => ({ name: name + '.png', format: 'PNG Image', size: 4, preview: '/textures/x.png' }));
    studio.run('allTextures = textures; renderTexturesGrid();');
    const grid = studio.elements.get('textures-grid');
    grid.children.at(-1).onclick();
    const cards = grid.children.filter(child => child.className === 'tex-card');
    assert.equal(cards.length, 81);
    return cards[0].innerHTML + cards[80].innerHTML;
}

test('hostile model and texture file names render as text, not as elements or attributes', () => {
    for (const render of [renderModelList, renderTextureGrid]) {
        const hostile = render(HOSTILE);
        assert.deepEqual(tagShape(hostile), tagShape(render('plain name')));
        assert.ok(visibleText(hostile).includes(HOSTILE), `${render.name} must still show the name`);
    }
});

test('a download that finishes or fails after a later click does not replace the model last clicked', () => {
    const studio = loadStudio();
    studio.run(`loadModel('/models/a.glb', 'a.glb', '', 1);
        loadModel('/models/b.glb', 'b.glb', '', 1);
        loadModel('/models/c.glb', 'c.glb', '', 1);`);
    const [a, b, c] = studio.loads;
    const modelC = fakeModel();
    c.onLoad({ scene: modelC });
    b.onError(new Error('late failure'));
    a.onLoad({ scene: fakeModel() });
    assert.equal(studio.run('currentMesh'), modelC);
    assert.deepEqual([...studio.run('scene.objects')], [modelC]);
    assert.equal(studio.elements.get('stat-name').innerText, 'c.glb');
    assert.deepEqual(studio.alerts, []);
});

test('switching models in clay mode disposes the old model\'s geometry, material and texture, not the clay', () => {
    const studio = loadStudio();
    const [modelA, modelB] = [fakeModel(), fakeModel()];
    studio.run(`setShadingMode('clay'); loadModel('/models/a.glb', 'a.glb', '', 1);`);
    studio.loads[0].onLoad({ scene: modelA });
    studio.run(`loadModel('/models/b.glb', 'b.glb', '', 1);`);
    studio.loads[1].onLoad({ scene: modelB });
    assert.deepEqual(modelA.parts.map(part => part.disposed), [true, true, true]);
    assert.deepEqual(modelB.parts.map(part => part.disposed), [false, false, false]);
    assert.equal(studio.run('clayMaterial.disposed'), false);
});

test('DDS inspector selects a mip and layer and clears controls for ordinary images', async () => {
    const { context, elements, run } = loadStudio();
    context.fetch = async () => ({ ok: true, json: async () => ({ width: 5, height: 3, mip_count: 3, array_size: 2 }) });
    await run("openModal({name:'a', preview:'/a.png', ddsPreview:'/api/dds/a.dds'})");
    const image = elements.get('modal-img');
    const controls = elements.get('dds-controls');
    const [mip, layer, dimensions] = controls.children;
    assert.equal(image.src, '/api/dds/a.dds?mip=0&layer=0');
    for (const label of [mip, layer]) {
        const input = label.children[0];
        Object.assign(input, {value: '1', valueAsNumber: 1, validity: {valid: true}});
        input.onchange();
    }
    assert.equal(image.src, '/api/dds/a.dds?mip=1&layer=1');
    assert.equal(dimensions.textContent, '2 \u00d7 1');
    await run("openModal({name:'b', preview:'/b.png'})");
    assert.equal(image.src, '/b.png');
    assert.equal(controls.children.length, 0);
});

test('volume slices shrink with the selected mip', async () => {
    const { context, elements, run } = loadStudio();
    const info = {dimension: '3d', width: 9, height: 3, depth: 5, mip_count: 3, array_size: 1};
    context.fetch = async () => ({ok: true, json: async () => info});
    await run("openModal({name:'v', preview:'/v.png', ddsPreview:'/api/dds/v.dds'})");
    const [mip, slice, dimensions] = elements.get('dds-controls').children;
    const z = slice.children[0];
    const image = elements.get('modal-img');
    function select(input, value) {
        Object.assign(input, {value: String(value), valueAsNumber: value, validity: {valid: true}});
        input.onchange();
    }
    assert.equal(z.max, 4);
    select(z, 4);
    assert.equal(image.src, '/api/dds/v.dds?mip=0&z=4');
    select(mip.children[0], 1);
    assert.equal(z.max, 1);
    assert.equal(z.value, 1);
    assert.equal(dimensions.textContent, '4 \u00d7 1');
    assert.equal(image.src, '/api/dds/v.dds?mip=1&z=1');
    select(mip.children[0], 2);
    assert.equal(z.disabled, true);
    assert.equal(image.src, '/api/dds/v.dds?mip=2&z=0');
    select(mip.children[0], 0);
    assert.equal(z.disabled, false);
});

test('half-float coordinates follow the selected mip and late reads cannot replace its value', async () => {
    const { context, elements, run } = loadStudio();
    const info = {format: 'R16_FLOAT', width: 5, height: 3, mip_count: 2, array_size: 1};
    const pending = [];
    context.fetch = url => url.endsWith('?info') ? {ok: true, json: async () => info} :
        new Promise(resolve => pending.push({url, resolve}));
    await run("openModal({name:'a', preview:'/a.png', ddsPreview:'/api/dds/a.dds'})");
    const [mip, x, y, output] = elements.get('dds-controls').children;
    x.children[0].value = 4;
    y.children[0].value = 2;
    const older = x.children[0].onchange();
    assert.match(pending[1].url, /mip=0&layer=0&x=4&y=2$/);
    Object.assign(mip.children[0], {value: '1', valueAsNumber: 1, validity: {valid: true}});
    mip.children[0].onchange();
    assert.match(pending[2].url, /mip=1&layer=0&x=1&y=0$/);
    pending[2].resolve({ok: true, json: async () => ({R: '-2.5'})});
    await new Promise(resolve => setImmediate(resolve));
    pending[1].resolve({ok: false});
    await older;
    assert.equal(output.textContent, 'R = -2.5');
});
