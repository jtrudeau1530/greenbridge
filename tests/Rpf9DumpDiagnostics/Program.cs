using System.Buffers.Binary;
using RpfPreview.Core;

Expect([], 4, "unrecognized-header");
Expect([], RageModelDecoder.MaxGraphicsBytes + 1, "graphics-size-limit");

// A synthetically truncated resource prefix containing a system-page pointer.
var prefix = new byte[16];
BinaryPrimitives.WriteUInt64LittleEndian(prefix.AsSpan(8), 0x50000080);
Expect(prefix, 16, "truncated-system");

// Recognizable drawable header with a deliberately malformed page table.
var header = new byte[0x100];
foreach (int at in new[] { 8, 16, 80, 168 })
    BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(at), 0x500000c0);
BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(44), 2);
foreach (int at in new[] { 64, 68, 72 }) BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(at), 1);
Expect(header, 16, "invalid-or-unsupported-layout");
Console.WriteLine("Model decode diagnostic smoke tests passed.");

// Util.WriteAtomic: outputs appear under their final name only once complete.
string dir = Directory.CreateTempSubdirectory("rpf9dump-test").FullName;
try
{
    string path = Path.Combine(dir, "a.dds");
    Util.WriteAllBytes(path, [1, 2, 3]);
    Check(File.ReadAllBytes(path).SequenceEqual(new byte[] { 1, 2, 3 }), "written file has the data");
    Check(Directory.GetFiles(dir).Length == 1, "no .partial left after a write");

    Util.WriteAllText(path, "new");
    Check(File.ReadAllText(path) == "new", "an existing file is replaced");

    string failed = Path.Combine(dir, "b.dds");
    try { Util.WriteAtomic(failed, t => { File.WriteAllBytes(t, [9]); throw new IOException("disk full"); }); Check(false, "error propagates"); }
    catch (IOException) { }
    Check(!File.Exists(failed) && !File.Exists(failed + ".partial"), "a failed write leaves neither file");

    // A stopped run's leftover .partial must not block a writer that opens with FileMode.CreateNew.
    string stale = Path.Combine(dir, "c.xma.wav");
    File.WriteAllBytes(stale + ".partial", [7]);
    Util.WriteAtomic(stale, t => { using var f = new FileStream(t, FileMode.CreateNew); f.WriteByte(5); });
    Check(File.ReadAllBytes(stale).SequenceEqual(new byte[] { 5 }) && !File.Exists(stale + ".partial"), "stale .partial is replaced");
}
finally { Directory.Delete(dir, true); }
Console.WriteLine("Atomic write tests passed.");

// Scan counts round-trip through the index cache's coverage text, including older culture-formatted caches.
Check(ScanCounts.Candidates(ScanCounts.Coverage(183801, 149811)) == 183801, "invariant coverage");
Check(ScanCounts.Candidates("rpf9dump full scan, 183,801 candidates, 149,811 recovered") == 183801, "comma-grouped coverage");
Check(ScanCounts.Candidates("rpf9dump full scan, 183\u00A0801 candidates, 149\u00A0811 recovered") == 183801, "space-grouped coverage");
Check(ScanCounts.Candidates("App scan of the first 16 MiB") is null, "coverage written by another tool");
Check(ScanCounts.Candidates(null) is null, "no cache");
Console.WriteLine("Scan count tests passed.");

static void Check(bool ok, string what)
{
    if (!ok) throw new Exception($"Failed: {what}");
}

static void Expect(ReadOnlySpan<byte> system, int graphicsBytes, string expected)
{
    string actual = ModelDecodeFailure.Explain(system, graphicsBytes).Code;
    if (actual != expected) throw new Exception($"Expected {expected}, got {actual}");
}
