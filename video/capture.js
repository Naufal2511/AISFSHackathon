// node video/capture.js <page.html> <out.mp4> <seconds> [fps]
// Loads the page, sets window.DUR, then calls window.render(t) for every frame and pipes PNGs to ffmpeg.
const puppeteer = require("puppeteer-core");
const { spawn } = require("child_process");
const path = require("path");
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
(async () => {
  const [page_, out, secs, fpsArg] = process.argv.slice(2);
  const fps = Number(fpsArg || 30), dur = Number(secs), frames = Math.round(dur * fps);
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new",
    args: ["--hide-scrollbars", "--force-device-scale-factor=1", "--allow-file-access-from-files"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });
  page.on("pageerror", e => console.error("page error:", e.message));
  await page.evaluateOnNewDocument(d => { window.DUR = d; }, dur);
  await page.goto("file://" + path.resolve(page_), { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(fps), "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", "-r", String(fps), out],
    { stdio: ["pipe", "inherit", "inherit"] });
  for (let i = 0; i < frames; i++) {
    await page.evaluate(t => window.render(t), i / fps);
    const buf = await page.screenshot({ type: "png" });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
  }
  ff.stdin.end();
  await new Promise(r => ff.on("close", r));
  console.log("wrote", out, frames, "frames");
  // Chrome sometimes never acknowledges close(); the video is already written, so don't wait on it
  await Promise.race([browser.close().catch(() => {}), new Promise(r => setTimeout(r, 3000))]);
  try { browser.process() && browser.process().kill("SIGKILL"); } catch (e) {}
  process.exit(0);
})();
