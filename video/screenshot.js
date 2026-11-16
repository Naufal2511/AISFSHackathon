// node video/screenshot.js <page.html> <out.png> [width] [fullpage:1|0] [selector]
const puppeteer = require("puppeteer-core");
const path = require("path");
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
(async () => {
  const [p, out, w, full, sel] = process.argv.slice(2);
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new", args: ["--allow-file-access-from-files", "--hide-scrollbars"] });
  const page = await browser.newPage();
  await page.setViewport({ width: Number(w || 1440), height: 900, deviceScaleFactor: 2 });
  await page.goto("file://" + path.resolve(p), { waitUntil: "networkidle0", timeout: 60000 }).catch(() => {});
  await new Promise(r => setTimeout(r, 800));
  if (sel) { const el = await page.$(sel); await el.screenshot({ path: out }); }
  else await page.screenshot({ path: out, fullPage: full !== "0" });
  console.log("wrote", out);
  await Promise.race([browser.close().catch(() => {}), new Promise(r => setTimeout(r, 3000))]);
  try { browser.process() && browser.process().kill("SIGKILL"); } catch (e) {}
  process.exit(0);
})();
