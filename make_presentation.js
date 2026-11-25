// Builds "Image Super-Resolution - Aditi Phulara.pptx"
//   NODE_PATH=<node_modules> DECK_MODULES=<node_modules> APPLY_THEME=<apply_theme.js> node make_presentation.js
const path = require("path");
const fs = require("fs");
const NM = process.env.DECK_MODULES || path.join(__dirname, "node_modules");
const pptxgen = require(path.join(NM, "pptxgenjs"));
const { applyTheme } = require(process.env.APPLY_THEME);

const IMG = (f) => path.join(__dirname, "slide_images", f);
const OUT = path.join(__dirname, "Image Super-Resolution - Aditi Phulara (v2).pptx");

// ---------- numbers straight from the result files ----------
const csv = (f) => {
  const [h, ...rows] = fs.readFileSync(path.join(__dirname, f), "utf8").trim().split(/\r?\n/);
  const keys = h.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [keys[i], v])));
};
const byMethod = (f) => Object.fromEntries(csv(f).map((r) => [r.method, r]));
const clean = byMethod("results/metrics.csv");
const real = byMethod("results_real/metrics.csv");
const extra = JSON.parse(fs.readFileSync(path.join(__dirname, "results/extra.json"), "utf8"));
clean["EDSR-REAL"] = { psnr_db: extra.edsr_real_clean.psnr, ssim: extra.edsr_real_clean.ssim };
const edsrLog = csv("runs/edsr_x4/log.csv");
const P = (t, m) => (+t[m].psnr_db).toFixed(2);
const S = (t, m) => (+t[m].ssim).toFixed(3);
const D = (a, b) => (a - b).toFixed(2);
const logLines = fs.readFileSync(path.join(__dirname, "train.log"), "utf8").split(/\r?\n/)
  .filter((l) => /^iter +(1000|10000|20000|30000) /.test(l) || /^done\. .*edsr_x4/.test(l))
  .slice(-5).map((l) => l.replace(/\s+\* new best/, "  *"));

// ---------- theme ----------
const THEME = {
  name: "Night Trails",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C2430", lt1: "FFFFFF", dk2: "0B1F3A", lt2: "F0F3F6",
    accent1: "D98E2B", accent2: "0B1F3A", accent3: "7A8696", accent4: "3E6E9E",
    accent5: "C8D3DF", accent6: "B5532E", hlink: "3E6E9E", folHlink: "6B5B95",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "Aditi Phulara";
pres.title = "Image Super-Resolution System";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "TITLE_DARK", background: { color: C.text2 }, objects: [] });
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.85,
      fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Image Super-Resolution System  |  Aditi Phulara", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});

// ---------- helpers ----------
const T = (slide, text, opts) => slide.addText(text, { isTextBox: true, fontSize: 17, color: C.text1, margin: 0, valign: "top", ...opts });
const pic = (slide, file, x, y, w, h, name) =>
  slide.addImage({ path: IMG(file), x, y, w, h, objectName: name, altText: name, line: { color: HEX.accent5, width: 0.75 } });
const cap = (slide, text, x, y, w, opts = {}) => T(slide, text, { x, y, w, h: 0.35, fontSize: 14, color: C.accent3, align: "center", ...opts });
const bullets = (items, o = {}) => items.map((t, i) => {
  const run = typeof t === "string" ? { text: t } : t;
  return { text: run.text, options: { bullet: o.num ? { type: "number" } : true, breakLine: i < items.length - 1, paraSpaceAfter: o.gap ?? 10, ...(run.options || {}) } };
});
const chartFont = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt" };
const barChart = (s, labels, values, colors, o) => s.addChart(pres.charts.BAR, [{ name: "PSNR (dB)", labels, values }], {
  barDir: "col", ...chartFont, showTitle: true, titleFontSize: 14, titleColor: HEX.dk2,
  chartColors: colors, invertedColors: colors,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 14, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.00",
  catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 14, valAxisLabelColor: HEX.accent3, valAxisLabelFontSize: 12,
  valAxisLabelFormatCode: "0.0", valGridLine: { color: "DDE3EA", size: 0.75 }, catGridLine: { style: "none" },
  showLegend: false, barGapWidthPct: 70, ...o,
});
const table = (s, rows, o) => s.addTable(rows.map((r, i) => r.map((c) => {
  const cell = typeof c === "object" ? c : { text: String(c) };
  return { text: cell.text, options: { fontSize: 15, color: HEX.dk1, valign: "middle", margin: [4, 8, 4, 8],
    fill: { color: i === 0 ? HEX.dk2 : (i % 2 ? HEX.lt1 : HEX.lt2) }, ...(i === 0 ? { color: HEX.lt1, bold: true } : {}), ...(cell.options || {}) } };
})), { border: { type: "solid", pt: 0.75, color: HEX.accent5 }, ...o });

(async () => {
  let s;
  // ============ Title ============
  pres.addSection({ title: "Introduction" });
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addImage({ path: IMG("hero.png"), x: 6.9, y: 0, w: 6.433, h: 7.5, sizing: { type: "cover", w: 6.433, h: 7.5 },
    objectName: "Before and after photo", altText: "Coastal photo, left half bicubic, right half EDSR" });
  s.addShape(pres.shapes.LINE, { x: 10.117, y: 0, w: 0, h: 7.5, line: { color: HEX.lt1, width: 1.5, dashType: "dash" }, objectName: "Split line" });
  for (const [x, t] of [[7.15, "bicubic"], [10.37, "my model"]]) {
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.3, w: 1.5, h: 0.4, fill: { color: C.text2, transparency: 25 }, line: { type: "none" }, objectName: t + " label" });
    T(s, t, { x, y: 0.3, w: 1.5, h: 0.4, fontSize: 13, color: C.background1, align: "center", valign: "middle" });
  }
  T(s, "Image Super-Resolution System", { x: 0.7, y: 1.9, w: 5.9, h: 1.7, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1 });
  T(s, "Reconstructing high-resolution images from low-resolution inputs using deep learning", { x: 0.7, y: 3.7, w: 5.6, h: 0.9, fontSize: 18, color: C.accent5 });
  T(s, [
    { text: "Aditi Phulara", options: { bold: true, fontSize: 18, color: C.background1, breakLine: true } },
    { text: "Department of Computer Science and Engineering", options: { fontSize: 14, color: C.accent5, breakLine: true } },
    { text: "Lovely Professional University", options: { fontSize: 14, color: C.accent5 } },
  ], { x: 0.7, y: 5.3, w: 5.9, h: 1.2 });
  s.addNotes("The photo on the right: left half is normal (bicubic) resizing, right half is my trained model, from the same small input.");

  // ============ What is SR ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("What is image super-resolution?", { placeholder: "title" });
  T(s, bullets([
    "Taking a small, low-resolution image and making a bigger, sharper version of it.",
    "A small image doesn't have enough pixels, so when we enlarge it the missing pixels have to be guessed.",
    "Normal resizing (bicubic) guesses by averaging nearby pixels. It never adds real detail, so edges come out blurry or jagged.",
    "A neural network that has seen thousands of sharp photos can make a much better guess.",
  ], { gap: 14 }), { x: 0.6, y: 1.5, w: 5.4, h: 4.8 });
  pic(s, "clean_lr_pixels.png", 6.4, 1.55, 3.1, 3.1, "Low-resolution input");
  pic(s, "clean_bicubic.png", 9.75, 1.55, 3.1, 3.1, "Bicubic result");
  cap(s, "Small input (64 × 64 px)", 6.4, 4.75, 3.1);
  cap(s, "Enlarged 4× with bicubic", 9.75, 4.75, 3.1);
  s.addNotes("Real crop from my test set. The thin light trails break into steps when you enlarge with bicubic.");

  // ============ Objectives ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Objectives", { placeholder: "title" });
  T(s, bullets([
    "Build a deep-learning model that upscales images by 4× (width and height).",
    "Compare it with bicubic interpolation and with a simpler network (SRCNN).",
    "Measure quality with PSNR and SSIM on test images the model has never seen.",
    "Make it work on real low-quality photos (blur, noise, JPEG), not only clean ones.",
    "Make a simple web page where anyone can upload an image and try it.",
  ], { num: true, gap: 16 }), { x: 0.6, y: 1.55, w: 7.2, h: 5.0 });
  // simple flow on the right
  const fy = [1.7, 3.55, 5.4];
  const ft = [["Low-resolution image", "e.g. 510 × 339"], ["Trained network", "EDSR / EDSR-real"], ["High-resolution image", "2040 × 1356"]];
  ft.forEach(([a, b], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 8.6, y: fy[i], w: 4.1, h: 1.15, fill: { color: i === 1 ? C.text2 : C.background2 }, line: { color: HEX.accent5, width: 0.75 }, objectName: a });
    T(s, [{ text: a, options: { bold: true, breakLine: true } }, { text: b, options: { fontSize: 14 } }],
      { x: 8.6, y: fy[i], w: 4.1, h: 1.15, align: "center", valign: "middle", color: i === 1 ? C.background1 : C.text1 });
    if (i < 2) s.addShape(pres.shapes.DOWN_ARROW, { x: 10.45, y: fy[i] + 1.2, w: 0.4, h: 0.6, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "Arrow" });
  });

  // ============ Tools ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Tools and setup", { placeholder: "title" });
  table(s, [
    ["", "What I used"],
    [{ text: "Language", options: { bold: true } }, "Python 3.13"],
    [{ text: "Deep learning", options: { bold: true } }, "PyTorch 2.14 with CUDA 12.6"],
    [{ text: "Hardware", options: { bold: true } }, "NVIDIA RTX 4050 Laptop GPU (6 GB)"],
    [{ text: "Dataset", options: { bold: true } }, "DIV2K (high-quality 2K photos)"],
    [{ text: "Image / maths", options: { bold: true } }, "Pillow, NumPy, scikit-image (for SSIM)"],
    [{ text: "Graphs", options: { bold: true } }, "Matplotlib"],
    [{ text: "Web demo", options: { bold: true } }, "Gradio"],
  ], { x: 0.6, y: 1.55, w: 7.4, colW: [2.4, 5.0], rowH: 0.55 });
  T(s, [
    { text: "Project files", options: { bold: true, color: C.text2, breakLine: true } },
    ...bullets(["train.py  (training)", "evaluate.py  (scores + figures)", "upscale.py  (one image)", "app.py  (web demo)", "sr/  (models, data, metrics)"], { gap: 6 }),
  ], { x: 8.6, y: 1.6, w: 4.2, h: 4.0, fontSize: 16, fontFace: "Calibri" });

  // ============ Dataset ============
  pres.addSection({ title: "Method" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Dataset: DIV2K", { placeholder: "title" });
  ["0812", "0823", "0828", "0811"].forEach((n, i) => pic(s, `thumb_${n}.png`, 0.6 + i * 3.1, 1.5, 2.85, 2.14, "DIV2K sample " + n));
  table(s, [
    ["Split", "Images", "Used for"],
    ["Training", "400", "Teaching the model"],
    ["Validation", "10", "Picking the best saved version"],
    ["Test", "90", "Final scores (never seen in training)"],
  ], { x: 0.6, y: 4.05, w: 7.0, colW: [1.7, 1.2, 4.1], rowH: 0.48 });
  T(s, "I don't need separate low-resolution photos: I shrink each sharp photo by 4× myself, so I always have the original to compare the output with.",
    { x: 8.0, y: 4.1, w: 4.75, h: 2.0, fontSize: 16 });
  s.addNotes("Full DIV2K has 800 training photos. The mirror I used only had the first 400, which was enough for about an hour of training.");

  // ============ Methods ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Methods compared", { placeholder: "title" });
  table(s, [
    ["Method", "Year", "How it works", "Parameters"],
    [{ text: "Bicubic", options: { bold: true } }, "–", "Fixed formula, averages the 16 nearest pixels. No learning. Used as the baseline.", "0"],
    [{ text: "SRCNN", options: { bold: true } }, "2014", "3 convolution layers. Enlarges with bicubic first, then learns to sharpen the result.", "69 K"],
    [{ text: "EDSR", options: { bold: true } }, "2017", "16 residual blocks, no batch-norm. Works on the small image and enlarges at the end with PixelShuffle.", "1.52 M"],
    [{ text: "EDSR-real", options: { bold: true } }, "mine", "Same as EDSR, fine-tuned on photos with added blur, noise and JPEG.", "1.52 M"],
  ], { x: 0.6, y: 1.55, w: 12.1, colW: [1.8, 1.0, 7.4, 1.9], rowH: 0.75 });
  T(s, "Parameters = the numbers the network learns during training.", { x: 0.6, y: 5.85, w: 12, h: 0.4, fontSize: 14, italic: true, color: C.accent3 });

  // ============ Code ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("The EDSR model in my code", { placeholder: "title" });
  const code = [
    "class ResBlock(nn.Module):",
    "    def forward(self, x):",
    "        return x + self.body(x)   # conv-relu-conv + skip",
    "",
    "class EDSR(nn.Module):",
    "    def forward(self, x):",
    "        x = x - self.mean           # normalise colours",
    "        x = self.head(x)            # 3 -> 64 channels",
    "        x = x + self.body(x)        # 16 ResBlocks",
    "        x = self.tail(x)            # PixelShuffle x2, x2",
    "        return x + self.mean",
  ].join("\n");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.5, w: 7.6, h: 3.6, fill: { color: "F4F6F8" }, line: { color: HEX.accent5, width: 0.75 }, objectName: "Code box" });
  T(s, code, { x: 0.85, y: 1.75, w: 7.2, h: 3.2, fontFace: "Courier New", fontSize: 14, color: C.text1 });
  T(s, "sr/models.py (shortened)", { x: 0.6, y: 5.2, w: 7.6, h: 0.35, fontSize: 13, italic: true, color: C.accent3 });
  T(s, bullets([
    { text: "Skip connections: each block only has to learn the change, which makes deep networks easier to train." },
    { text: "Works at low resolution, so it's faster than doing all 16 blocks on the big image." },
    { text: "PixelShuffle rearranges 4 channels into a 2× bigger image; done twice for 4×." },
  ], { gap: 14 }), { x: 8.6, y: 1.55, w: 4.2, h: 4.8, fontSize: 16 });

  // ============ Training ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Training", { placeholder: "title" });
  T(s, bullets([
    "Cut a random 192 × 192 piece out of a training photo",
    "Shrink it to 48 × 48 (this is the input)",
    "Model enlarges it back to 192 × 192",
    "Compare with the real piece (L1 loss) and update the weights",
  ], { num: true, gap: 8 }), { x: 0.6, y: 1.5, w: 5.8, h: 2.4, fontSize: 16 });
  T(s, "Batch of 16, 30,000 steps, Adam optimiser, about 39 minutes.", { x: 0.6, y: 3.9, w: 5.8, h: 0.4, fontSize: 15, italic: true, color: C.accent3 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.45, w: 5.8, h: 1.65, fill: { color: "1E1E1E" }, line: { type: "none" }, objectName: "Terminal" });
  T(s, ["> python train.py --model edsr --iters 30000", ...logLines].join("\n"),
    { x: 0.75, y: 4.58, w: 5.55, h: 1.45, fontFace: "Courier New", fontSize: 10.5, color: "D4D4D4" });
  s.addChart(pres.charts.LINE, [{ name: "Validation PSNR (dB)", labels: edsrLog.map((r) => (r.iter % 5000 === 0 ? r.iter / 1000 + "k" : "")), values: edsrLog.map((r) => +r.val_psnr) }], {
    x: 6.8, y: 1.45, w: 6.1, h: 5.3, ...chartFont,
    showTitle: true, title: "Validation PSNR (dB) during EDSR training", titleFontSize: 14, titleColor: HEX.dk2,
    chartColors: [HEX.accent1], lineSize: 3, lineDataSymbol: "none",
    catAxisLabelColor: HEX.accent3, valAxisLabelColor: HEX.accent3, catAxisLabelFontSize: 11, valAxisLabelFontSize: 12,
    catAxisLabelFrequency: 1, valAxisMinVal: 29.5, valAxisMaxVal: 31, valAxisMajorUnit: 0.5, valAxisLabelFormatCode: "0.0",
    valGridLine: { color: "DDE3EA", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false, objectName: "Training curve",
  });
  s.addNotes("The terminal lines are copied from my actual training log. The curve flattens near the end, so longer training would only help a little.");

  // ============ Metrics ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("How I measured quality", { placeholder: "title" });
  T(s, "PSNR (Peak Signal-to-Noise Ratio)", { x: 0.6, y: 1.5, w: 5.9, h: 0.45, bold: true, color: C.text2, fontSize: 19 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 2.05, w: 5.9, h: 0.8, fill: { color: C.background2 }, line: { type: "none" }, objectName: "Formula box" });
  T(s, "PSNR = 10 · log10( 255² / MSE )", { x: 0.6, y: 2.05, w: 5.9, h: 0.8, fontFace: "Cambria", fontSize: 20, align: "center", valign: "middle" });
  T(s, bullets([
    "MSE = average squared difference between output and original pixels",
    "Measured in dB. Higher is better",
    "+1 dB is already a visible improvement",
  ], { gap: 8 }), { x: 0.6, y: 3.05, w: 5.9, h: 2.0, fontSize: 16 });
  T(s, "SSIM (Structural Similarity)", { x: 7.0, y: 1.5, w: 5.8, h: 0.45, bold: true, color: C.text2, fontSize: 19 });
  T(s, bullets([
    "Checks if edges, textures and contrast look the same as in the original",
    "Score from 0 to 1. Closer to 1 is better",
    "Closer to how people judge images than PSNR",
  ], { gap: 8 }), { x: 7.0, y: 2.1, w: 5.8, h: 2.4, fontSize: 16 });
  T(s, [
    { text: "Why no accuracy or confusion matrix? ", options: { bold: true } },
    { text: "This is not classification. The output is an image, so we measure how close it is to the original. All super-resolution papers use PSNR and SSIM. Both are computed on the brightness (Y) channel." },
  ], { x: 0.6, y: 5.3, w: 12.1, h: 1.0, fontSize: 15, color: C.text1 });

  // ============ Results clean ============
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Results on 90 test images (×4)", { placeholder: "title" });
  barChart(s, ["Bicubic", "SRCNN", "EDSR"], [P(clean, "Bicubic"), P(clean, "SRCNN"), P(clean, "EDSR")].map(Number),
    [HEX.accent3, HEX.accent4, HEX.accent1], { x: 0.6, y: 1.45, w: 6.6, h: 5.3, title: "Average PSNR (dB)", valAxisMinVal: 27, valAxisMaxVal: 30.5, valAxisMajorUnit: 0.5, objectName: "PSNR chart" });
  const bold = (t) => ({ text: t, options: { bold: true } });
  table(s, [
    ["Method", "PSNR (dB)", "SSIM"],
    ["Bicubic", P(clean, "Bicubic"), S(clean, "Bicubic")],
    ["SRCNN", P(clean, "SRCNN"), S(clean, "SRCNN")],
    [bold("EDSR"), bold(P(clean, "EDSR")), bold(S(clean, "EDSR"))],
  ], { x: 7.6, y: 1.6, w: 5.1, colW: [1.9, 1.6, 1.6], rowH: 0.5, align: "center" });
  T(s, bullets([
    `EDSR is ${D(clean.EDSR.psnr_db, clean.Bicubic.psnr_db)} dB better than bicubic and about 1 dB better than SRCNN.`,
    "So the deeper network really helps.",
    "Inputs here were made by clean bicubic shrinking (the standard test setup).",
  ], { gap: 10 }), { x: 7.6, y: 3.9, w: 5.1, h: 2.6, fontSize: 16 });

  // ============ Visual ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Visual comparison", { placeholder: "title" });
  [["clean_bicubic.png", "Bicubic"], ["clean_srcnn.png", "SRCNN"], ["clean_edsr.png", "EDSR"], ["clean_hr.png", "Original"]].forEach(([f, l], i) => {
    const x = 0.6 + i * 3.075;
    pic(s, f, x, 1.5, 2.85, 2.85, l);
    cap(s, l, x, 4.45, 2.85, { fontSize: 16, color: C.text1, bold: l === "EDSR" });
  });
  T(s, "Same crop of a test image, same 64 × 64 input to every method. Bicubic breaks the light trails into steps, SRCNN is a bit better, EDSR gives clean continuous lines close to the original.",
    { x: 0.6, y: 5.15, w: 12.1, h: 1.0, fontSize: 16 });
  s.addNotes("I picked this crop because the difference is easy to see. The averages over all 90 images are on the previous slide.");

  // ============ Problem with real photos ============
  pres.addSection({ title: "Real-world photos" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Real-world photos" });
  s.addText("Problem: real photos are not clean", { placeholder: "title" });
  T(s, bullets([
    "Real low-quality photos also have blur, camera noise and JPEG blocks.",
    "EDSR only ever saw clean images, so it treats the noise as detail and sharpens it.",
    { text: `On damaged test photos EDSR scored ${D(real.Bicubic.psnr_db, real.EDSR.psnr_db)} dB worse than plain bicubic.`, options: { bold: true, color: C.accent6 } },
  ], { gap: 14 }), { x: 0.6, y: 1.5, w: 5.0, h: 4.6 });
  pic(s, "real_input.png", 6.1, 1.55, 3.25, 3.25, "Damaged input");
  pic(s, "real_edsr.png", 9.6, 1.55, 3.25, 3.25, "EDSR output");
  cap(s, "Damaged input (blur + noise + JPEG)", 6.1, 4.9, 3.25);
  cap(s, "EDSR output (scratchy artefacts)", 9.6, 4.9, 3.25);

  // ============ Fix ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Real-world photos" });
  s.addText("Solution: EDSR-real", { placeholder: "title" });
  T(s, "I took the trained EDSR and trained it for 12,000 more steps on photos I damaged on purpose. Each time the damage is random: blur, shrink, camera noise, JPEG.",
    { x: 0.6, y: 1.45, w: 6.0, h: 1.2, fontSize: 16 });
  [["damage_0.png", "clean"], ["damage_1.png", "random 1"], ["damage_2.png", "random 2"], ["damage_3.png", "random 3"]].forEach(([f, l], i) => {
    const x = 0.6 + (i % 2) * 3.0, y = 2.8 + Math.floor(i / 2) * 2.05;
    pic(s, f, x, y, 1.65, 1.65, "Training input " + l);
    T(s, l, { x: x + 1.8, y: y + 0.6, w: 1.1, h: 0.4, fontSize: 14, color: C.accent3 });
  });
  barChart(s, ["Bicubic", "SRCNN", "EDSR", "EDSR-real"], ["Bicubic", "SRCNN", "EDSR", "EDSR-REAL"].map((m) => +P(real, m)),
    [HEX.accent3, HEX.accent4, HEX.accent4, HEX.accent1], { x: 6.9, y: 1.45, w: 6.0, h: 5.3, title: "PSNR on 90 damaged test images (dB)", valAxisMinVal: 25, valAxisMaxVal: 27.5, valAxisMajorUnit: 0.5, objectName: "PSNR chart real" });
  s.addNotes(`EDSR-real: ${P(real, "EDSR-REAL")} dB, ${D(real["EDSR-REAL"].psnr_db, real.Bicubic.psnr_db)} dB above bicubic and ${D(real["EDSR-REAL"].psnr_db, real.EDSR.psnr_db)} dB above EDSR. SSIM ${S(real, "EDSR-REAL")}. Downside: about 1 dB lower than EDSR on perfectly clean images.`);

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Real-world photos" });
  s.addText("EDSR vs EDSR-real on a damaged image", { placeholder: "title" });
  [["real_input.png", "Damaged input"], ["real_edsr.png", "EDSR"], ["real_edsr_real.png", "EDSR-real"], ["real_hr.png", "Original"]].forEach(([f, l], i) => {
    const x = 0.6 + i * 3.075;
    pic(s, f, x, 1.5, 2.85, 2.85, l);
    cap(s, l, x, 4.45, 2.85, { fontSize: 16, color: C.text1, bold: l === "EDSR-real" });
  });
  T(s, "EDSR-real removes the noise and JPEG blocks instead of sharpening them. It is a bit smoother than the original, which is the trade-off of training with a simple L1 loss.",
    { x: 0.6, y: 5.15, w: 12.1, h: 1.0, fontSize: 16 });

  // ============ Demo ============
  pres.addSection({ title: "Wrap-up" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
  s.addText("Web demo (Gradio)", { placeholder: "title" });
  pic(s, "demo_ui_crop.png", 0.6, 1.45, 7.6, 4.886, "Screenshot of the Gradio demo");
  T(s, [
    { text: "Upload a photo, choose a model and click Run. It shows bicubic and the model output next to each other.", options: { breakLine: true, paraSpaceAfter: 14 } },
    { text: "Modes:", options: { bold: true, breakLine: true } },
    ...bullets(["Upscale my image", "Simulate clean low-res (shows PSNR)", "Simulate real-world low-res"], { gap: 6 }),
  ], { x: 8.6, y: 1.5, w: 4.2, h: 3.8, fontSize: 16 });
  T(s, "Run with:  python app.py", { x: 8.6, y: 5.5, w: 4.2, h: 0.4, fontFace: "Courier New", fontSize: 14 });

  // ============ Problems ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
  s.addText("Problems I ran into", { placeholder: "title" });
  table(s, [
    ["Problem", "What I did"],
    ["The official DIV2K server downloaded at only ~60 KB/s", "Switched to a Hugging Face mirror of the same files"],
    ["That mirror only had 400 of the 800 training images", "Trained on 400 (enough for ~1 hour of training)"],
    ["EDSR got worse than bicubic on noisy photos", "Built the degradation pipeline and trained EDSR-real"],
    ["SRCNN was almost as slow as EDSR (276 vs 329 ms per image)", "Found out why: SRCNN runs on the full-size image, EDSR on the small one"],
  ], { x: 0.6, y: 1.55, w: 12.1, colW: [5.8, 6.3], rowH: 0.75 });

  // ============ Limitations ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
  s.addText("Limitations and future work", { placeholder: "title" });
  T(s, "Limitations", { x: 0.6, y: 1.5, w: 5.8, h: 0.45, bold: true, color: C.text2, fontSize: 19 });
  T(s, bullets([
    "Only 400 training photos and about an hour of training",
    "Only 4× upscaling",
    "No faces, text or cartoons in the training data",
    "EDSR-real outputs are a little smooth",
    "Detail that is completely lost can't be recovered",
  ]), { x: 0.6, y: 2.1, w: 5.8, h: 4.2, fontSize: 16 });
  T(s, "Future work", { x: 7.0, y: 1.5, w: 5.8, h: 0.45, bold: true, color: C.text2, fontSize: 19 });
  T(s, bullets([
    "Train longer on all 800 DIV2K + Flickr2K images",
    "Add 2× and 3× models (code already supports it)",
    "Add a GAN / perceptual loss (like ESRGAN) for sharper textures",
    "Test on Set5, Set14, BSD100, Urban100",
  ]), { x: 7.0, y: 2.1, w: 5.8, h: 4.2, fontSize: 16 });

  // ============ Conclusion ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
  s.addText("Conclusion", { placeholder: "title" });
  T(s, bullets([
    `I built a working super-resolution system: training, testing, and a web demo.`,
    `On clean images, EDSR beats bicubic by ${D(clean.EDSR.psnr_db, clean.Bicubic.psnr_db)} dB PSNR (SSIM ${S(clean, "Bicubic")} → ${S(clean, "EDSR")}).`,
    `On real damaged photos, plain EDSR was worse than bicubic, but EDSR-real was ${D(real["EDSR-REAL"].psnr_db, real.Bicubic.psnr_db)} dB better.`,
    "Main thing I learned: the training data matters as much as the model.",
  ], { gap: 18 }), { x: 0.6, y: 1.6, w: 7.4, h: 4.8, fontSize: 18 });
  pic(s, "clean_edsr.png", 8.6, 1.6, 4.1, 4.1, "EDSR output");
  cap(s, "EDSR output on a test image", 8.6, 5.8, 4.1);

  // ============ References ============
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
  s.addText("References", { placeholder: "title" });
  T(s, bullets([
    "C. Dong, C. C. Loy, K. He and X. Tang, “Image super-resolution using deep convolutional networks,” IEEE TPAMI, 2016. (SRCNN)",
    "B. Lim, S. Son, H. Kim, S. Nah and K. M. Lee, “Enhanced deep residual networks for single image super-resolution,” CVPRW, 2017. (EDSR)",
    "W. Shi et al., “Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network,” CVPR, 2016. (PixelShuffle)",
    "X. Wang, L. Xie, C. Dong and Y. Shan, “Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data,” ICCVW, 2021.",
    "Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE TIP, 2004. (SSIM)",
    "E. Agustsson and R. Timofte, “NTIRE 2017 challenge on single image super-resolution: dataset and study,” CVPRW, 2017. (DIV2K)",
  ], { num: true, gap: 10 }), { x: 0.6, y: 1.5, w: 12.1, h: 5.2, fontSize: 15 });

  // ============ Thank you ============
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Wrap-up" });
  T(s, "Thank you", { x: 0.7, y: 2.6, w: 11.9, h: 1.2, fontFace: THEME.headFontFace, fontSize: 48, bold: true, color: C.background1, align: "center" });
  T(s, "Questions?", { x: 0.7, y: 3.8, w: 11.9, h: 0.6, fontSize: 22, color: C.accent5, align: "center" });
  T(s, "Aditi Phulara", { x: 0.7, y: 5.6, w: 11.9, h: 0.5, fontSize: 16, color: C.accent5, align: "center" });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
