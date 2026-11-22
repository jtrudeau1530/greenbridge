// Bulk exporter around rpf9-explorer's Core (github.com/breadbbbb/rpf9-explorer).
// Local use only: writes OBJ/DDS/JSON from the user's own package. No keys, no TOC decryption.
using System.Diagnostics;
using System.Text.Json;
using RpfPreview.Core;

if (args.Length < 3) { Console.WriteLine("usage: rpf9dump ttpf|models|textures <source> <outdir>"); return 2; }
string mode = args[0], src = Path.GetFullPath(args[1]), outRoot = Path.GetFullPath(args[2]);
var json = new JsonSerializerOptions { WriteIndented = true };
var sw = Stopwatch.StartNew();
string baseName = Path.GetFileNameWithoutExtension(src);
string outDir = Path.Combine(outRoot, baseName); Directory.CreateDirectory(outDir);
long length = new FileInfo(src).Length;
int ok = 0, fail = 0, skip = 0;

if (mode == "ttpf")
{
    var info = TtpfReader.Inspect(src);
    var rows = new List<object>();
    foreach (var t in info.Textures)
    {
        string name = $"{t.Index:D6}_{Util.Safe(Path.GetFileNameWithoutExtension(t.Name))}.dds";
        string path = Path.Combine(outDir, name);
        string? file = t.CanExtract ? name : null, limitation = t.Limitation;
        if (!t.CanExtract) skip++;
        else if (File.Exists(path)) ok++;
        else
        {
            try
            {
                // Content the decoder cannot handle is a skip like !CanExtract. Pool read errors, DDS write errors,
                // cancellation and anything unexpected stay failures.
                byte[]? dds = null;
                try { dds = TtpfReader.CreateDds(src, t.Index); }
                catch (Exception e) when (e.GetType() == typeof(NotSupportedException) || e.GetType() == typeof(InvalidDataException))
                { skip++; file = null; limitation = e.Message; }
                if (dds != null) { Util.WriteAllBytes(path, dds); ok++; }
            }
            catch (Exception e) { fail++; if (fail <= 5) Console.WriteLine($"  fail {t.Index}: {e.GetType().Name} {e.Message}"); }
            if ((ok + fail) % 5000 == 0) Console.WriteLine($"  {ok + fail}/{info.Textures.Count}");
        }
        rows.Add(new { t.Index, t.Name, t.Id, AssetKey = t.AssetKey.ToString("X16"), t.Width, t.Height, t.DxgiFormat, t.Mips, t.SourceMips, t.BaseMipLevel, Limitation = limitation, File = file });
    }
    // The manifest is the build's completion marker. Keep successful DDS for retry, but withdraw a stale marker.
    string manifest = Path.Combine(outDir, "manifest.json");
    if (fail == 0)
        Util.WriteAllText(manifest, JsonSerializer.Serialize(new { src, info.Version, Count = info.Textures.Count, info.Metadata, Textures = rows }, json));
    else
        File.Delete(manifest);
}
else if (mode == "audio")
{
    if (!AwcRecovery.HasCompleteCache(src))
    {
        long last = -1;
        var prog = new Progress<long>(p => { long s = p / (4L << 30); if (s != last) { last = s; Console.WriteLine($"  awc scan {100.0 * p / length:0.0}%"); } });
        var found = AwcRecovery.RecoverFull(src, prog);
        AwcRecovery.WriteCache(src, found);
        Console.WriteLine($"  {found.Count:N0} AWC banks ({sw.Elapsed.TotalSeconds:0}s)");
    }
    var doc = ArchiveReader.Open(src);
    var banks = doc.Entries.Where(e => e.Extension == ".awc" && e.CanExtract).ToList();
    string sub = Path.Combine(outDir, "awc");
    var todo = banks.Where(e => !File.Exists(ArchiveReader.SafeDestination(sub, e.Path))).ToList();
    Console.WriteLine($"  {banks.Count:N0} banks, {todo.Count:N0} to write");
    for (int i = 0; i < todo.Count; i += 500)
        doc.ExtractAsync(todo.Skip(i).Take(500), sub).GetAwaiter().GetResult();
    ok = banks.Count;
    var rows = banks.Select(e => { var info = AwcInfo.Inspect(src, e.Offset, e.Size);
        return new { e.Path, e.Offset, e.Size, Streams = info?.Streams.Select(s => new { s.Index, Id = s.Id.ToString("X8"), s.Codec, s.Channels, s.SampleRate, s.DurationSeconds, s.IsStreamed }) }; }).ToList();
    Util.WriteAllText(Path.Combine(outDir, "audio_manifest.json"), JsonSerializer.Serialize(new { src, Count = banks.Count, Banks = rows }, json));
}
else if (mode == "xma")
{
    // XMA2 streams (AWC codec 5) as RIFF XMA2 .wav files that ffmpeg decodes (AwcRecovery.WriteXmaWave).
    var doc = ArchiveReader.Open(src);
    var banks = doc.Entries.Where(e => e.Extension == ".awc" && e.CanExtract).ToList();
    string sub = Path.Combine(outDir, "xma"); Directory.CreateDirectory(sub);
    foreach (var e in banks)
    {
        var info = AwcInfo.Inspect(src, e.Offset, e.Size);
        if (info == null) continue;
        foreach (var st in info.Streams.Where(x => x.Codec == 5))
        {
            string dir = Path.Combine(sub, Util.Safe(Path.GetFileNameWithoutExtension(e.Name)));
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, $"{st.Id:X8}.xma.wav");
            if (File.Exists(path)) { ok++; continue; }
            try { Util.WriteAtomic(path, tmp => AwcRecovery.WriteXmaWave(src, e.Offset, e.Size, st.Index, tmp)); ok++; }
            catch (Exception ex) { fail++; if (fail <= 5) Console.WriteLine($"  fail {e.Name}#{st.Index}: {ex.GetType().Name} {ex.Message}"); }
        }
    }
}
else if (mode is "models" or "textures")
{
    byte[] header; using (var f = File.OpenRead(src)) { header = new byte[64]; f.ReadExactly(header); }
    var cached = RecoveryIndexCache.Load(mode, src, length, header);
    if (cached == null)
    {
        long last = -1;
        var prog = new Progress<long>(p => { long s = p / (4L << 30); if (s != last) { last = s; Console.WriteLine($"  scan {100.0 * p / length:0.0}%"); } });
        var cands = mode == "models" ? Rpf9ModelRecovery.FindCandidateOffsets(src, progress: prog) : Rpf9TextureRecovery.FindCandidateOffsets(src, progress: prog);
        Console.WriteLine($"  candidates {cands.Count:N0} ({sw.Elapsed.TotalSeconds:0}s)");
        var entries = new List<ArchiveEntry>();
        for (int i = 0; i < cands.Count; i += 32)
        {
            var chunk = cands.Skip(i).Take(32).ToArray();
            var got = mode == "models" ? Rpf9ModelRecovery.RecoverAtOffsets(src, length, chunk, entries.Count) : Rpf9TextureRecovery.RecoverAtOffsets(src, length, chunk, entries.Count);
            entries.AddRange(got);
            if ((i / 32) % 200 == 0) Console.WriteLine($"  validated {i + chunk.Length:N0}/{cands.Count:N0} -> {entries.Count:N0} ({sw.Elapsed.TotalSeconds:0}s)");
        }
        entries = entries.OrderBy(e => e.Offset).Select((e, k) => e with { Index = k }).ToList();
        RecoveryIndexCache.Save(mode, src, entries, $"rpf9dump full scan, {cands.Count:N0} candidates, {entries.Count:N0} recovered");
    }
    var doc = ArchiveReader.Open(src);
    string want = mode == "models" ? "Recovered model" : "Recovered texture dictionary";
    var list = doc.Entries.Where(e => e.Compression == want).ToList();
    Console.WriteLine($"  {list.Count:N0} {mode} entries ({sw.Elapsed.TotalSeconds:0}s)");
    string sub = Path.Combine(outDir, mode); Directory.CreateDirectory(sub);
    // The manifest is streamed row by row: holding 150k rows (each capturing its decoded model through lazy
    // projections) ran out of memory on levels.rpf. Written to a .partial file and renamed when complete.
    string manifestPath = Path.Combine(sub, "manifest.json"), partial = manifestPath + ".partial";
    using var mstream = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
    using var writer = new Utf8JsonWriter(mstream, new JsonWriterOptions { Indented = true });
    writer.WriteStartObject();
    writer.WriteString("src", src);
    writer.WriteNumber("Count", list.Count);
    writer.WriteStartArray("Items");
    void Row(object row) => JsonSerializer.Serialize(writer, row, json);
    int embedded = 0;
    var modelDecodeFailures = new SortedDictionary<string, int>(StringComparer.Ordinal);
    foreach (var e in list)
    {
        string stem = Util.Safe(Path.GetFileNameWithoutExtension(e.Name));
        try
        {
            var data = ArchiveReader.ReadResourceData(doc, e);
            int split = e.ResourceSystemBytes ?? checked((int)e.Segments![0].Size);
            var sys = data.AsSpan(0, split); var gfx = data.AsSpan(split);
            if (mode == "models")
            {
                var m = RageModelDecoder.Decode(sys, gfx, e.ResourceDrawableOffset);
                if (m == null)
                {
                    var reason = ModelDecodeFailure.Explain(sys, gfx.Length);
                    int count = modelDecodeFailures.GetValueOrDefault(reason.Code) + 1;
                    modelDecodeFailures[reason.Code] = count;
                    if (count <= 3)
                        Console.Error.WriteLine($"  model decode failed at 0x{e.Offset:X}: {reason.Code} ({reason.Detail})");
                    fail++;
                    continue;
                }
                string obj = Path.Combine(sub, stem + ".obj");
                if (!File.Exists(obj)) Util.WriteAllText(obj, RageModelDecoder.CreateObj(m));
                if (m.EmbeddedTextures != null && m.EmbeddedTextureDictionaryOffset is int dOff)
                {
                    string td = Path.Combine(sub, stem + "_tex"); Directory.CreateDirectory(td);
                    foreach (var t in m.EmbeddedTextures)
                    {
                        string p = Path.Combine(td, $"{t.Index:D3}_{Util.Safe(t.Name)}.dds");
                        if (File.Exists(p)) { embedded++; continue; }
                        try { Util.WriteAllBytes(p, Util.BgraDds(RageTextureDecoder.DecodeEmbeddedTexturePixels(sys, gfx, dOff, t.Index))); embedded++; } catch { }
                    }
                }
                Row(new { e.Offset, Drawable = e.ResourceDrawableOffset, File = stem + ".obj", m.Name, Meshes = m.Meshes.Count, Verts = m.Meshes.Sum(x => x.Positions.Count), Tris = m.Meshes.Sum(x => x.Indices.Count) / 3, m.IsFragment, m.HasBoneBindings,
                    Min = new[] { m.BoundsMin.X, m.BoundsMin.Y, m.BoundsMin.Z }, Max = new[] { m.BoundsMax.X, m.BoundsMax.Y, m.BoundsMax.Z },
                    MeshShaders = m.Meshes.Select(x => x.ShaderIndex).ToArray(),
                    Embedded = m.EmbeddedTextures?.Select(t => new { t.Index, t.Name, t.Width, t.Height, t.Format, Hash = t.NameHash.ToString("X8"), Key = t.AssetKey?.ToString("X16") }).ToArray(),
                    Materials = m.Materials?.Select(x => new { x.ShaderIndex, Shader = x.ShaderHash.ToString("X8"), Preset = x.PresetHash.ToString("X8"), x.RenderBucket,
                        Textures = x.Textures.Select(r => new { Param = r.ParameterHash.ToString("X8"), r.TextureIndex, Name = r.NameHash.ToString("X8"), Id64 = r.OpaqueId64.ToString("X16"), Tag = r.Tag.ToString("X8") }).ToArray() }).ToArray() });
                ok++;
            }
            else
            {
                var dict = RageTextureDecoder.Inspect(sys, gfx);
                if (dict == null) { fail++; continue; }
                string td = Path.Combine(sub, stem); Directory.CreateDirectory(td);
                foreach (var t in dict.Textures)
                {
                    string p = Path.Combine(td, $"{t.Index:D3}_{Util.Safe(t.Name)}.dds");
                    try { if (!File.Exists(p)) Util.WriteAllBytes(p, RageTextureDecoder.CreateDds(sys, gfx, t.Index)); ok++; } catch { fail++; }
                }
                Row(new { e.Offset, Dir = stem, Textures = dict.Textures.Select(t => new { t.Index, t.Name, t.Width, t.Height, t.Format, t.MipCount, Hash = t.NameHash.ToString("X8"), Key = t.AssetKey?.ToString("X16") }).ToArray() });
            }
        }
        catch (Exception ex) { fail++; if (fail <= 5) Console.WriteLine($"  fail {e.Name}: {ex.GetType().Name} {ex.Message}"); }
        if ((ok + fail) % 1000 == 0) Console.WriteLine($"  exported {ok + fail:N0}/{list.Count:N0}");
    }
    if (mode == "models") Console.WriteLine($"  embedded textures {embedded:N0}");
    writer.WriteEndArray();
    if (mode == "models")
    {
        writer.WritePropertyName("ModelDecodeFailures");
        JsonSerializer.Serialize(writer, modelDecodeFailures, json);
        foreach (var (code, count) in modelDecodeFailures)
            Console.Error.WriteLine($"  model decode failures ({code}): {count:N0}");
    }
    writer.WriteEndObject();
    writer.Dispose();          // flushes; the later using-dispose is a no-op
    mstream.Dispose();
    File.Move(partial, manifestPath, true);
}
Console.WriteLine($"{mode} {baseName}: ok={ok} fail={fail} skip={skip} in {sw.Elapsed.TotalSeconds:0}s");
return mode == "ttpf" && fail != 0 ? 1 : 0;
