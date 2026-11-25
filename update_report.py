"""Update the unpacked IEEE report (word/document.xml) so it matches this project.
Only the text of factually wrong paragraphs / table cells is replaced; formatting is kept.

    python update_report.py <unpacked_dir>
"""
import csv
import json
import re
import sys
from xml.sax.saxutils import escape

D = sys.argv[1]
DOC = f"{D}/word/document.xml"

# ---------------- measured numbers ----------------
def rows(path):
    return {r["method"]: r for r in csv.DictReader(open(path))}

clean, real = rows("results/metrics.csv"), rows("results_real/metrics.csv")
extra = json.load(open("results/extra.json"))
t = extra["timing_ms"]
er = extra["edsr_real_clean"]
clean["EDSR-REAL"] = {"psnr_db": f"{er['psnr']:.3f}", "ssim": f"{er['ssim']:.4f}"}

P = lambda tab, m: float(tab[m]["psnr_db"])
S = lambda tab, m: float(tab[m]["ssim"])
d = lambda v: f"{v:+.3f}".replace("-", "−")
d4 = lambda v: f"{v:+.4f}".replace("-", "−")

g_edsr = P(clean, "EDSR") - P(clean, "Bicubic")
g_edsr_srcnn = P(clean, "EDSR") - P(clean, "SRCNN")
g_srcnn = P(clean, "SRCNN") - P(clean, "Bicubic")
r_srcnn = P(real, "SRCNN") - P(real, "Bicubic")
r_edsr = P(real, "EDSR") - P(real, "Bicubic")
r_real = P(real, "EDSR-REAL") - P(real, "Bicubic")
r_real_edsr = P(real, "EDSR-REAL") - P(real, "EDSR")
c_real_cost = P(clean, "EDSR") - P(clean, "EDSR-REAL")
ms = {"Bicubic": t["Bicubic"], "SRCNN": t["SRCNN"], "EDSR": t["EDSR"], "EDSR-REAL": t["EDSR-real"]}
f1 = lambda v: f"{v:.1f}"

# ---------------- new text, by paragraph index ----------------
NEW = {
    7: ("This paper describes an end-to-end image super-resolution (SR) system that reconstructs ×4 "
        "high-resolution images from low-resolution inputs. Two deep-learning architectures, SRCNN and EDSR, "
        "are trained on the DIV2K dataset with a shared PyTorch training and evaluation pipeline, and a Gradio "
        "web interface and a command-line tool provide inference on arbitrary images. To handle realistic "
        "inputs, EDSR is additionally fine-tuned on a randomized degradation model of blur, down-sampling, "
        "noise and JPEG compression (EDSR-real). All models were trained on a single laptop GPU and evaluated "
        "on 90 held-out DIV2K images using PSNR and SSIM on the luminance channel. On clean bicubic inputs "
        f"EDSR reaches {P(clean,'EDSR'):.2f} dB PSNR-Y, a gain of {g_edsr:.2f} dB over bicubic interpolation "
        f"and {g_edsr_srcnn:.2f} dB over SRCNN. On degraded inputs the bicubic-trained models fall below the "
        f"bicubic baseline, whereas EDSR-real reaches {P(real,'EDSR-REAL'):.2f} dB, a gain of {r_real:.2f} dB. "
        "The work does not claim architectural novelty; its contribution is a reproducible, fully executable "
        "system and an analysis of the effect of the training degradation."),
    8: "image super-resolution, SRCNN, EDSR, real-world degradation, PSNR, SSIM, DIV2K, deep learning.",
    11: ("This work presents a single repository that integrates (i) two SR architectures at ×4, (ii) a data "
         "pipeline that generates LR inputs on the fly from DIV2K HR images, including a real-world degradation "
         "model, (iii) training with mixed precision, cosine learning-rate decay and best-checkpoint selection, "
         "(iv) an evaluation suite reporting PSNR and SSIM, (v) a command-line inference tool, and (vi) a web "
         "demonstration interface. A guiding principle is that every component executes: there is no "
         "placeholder inference and no fabricated metric, and every reported number is produced by running a "
         "trained checkpoint over a held-out test set."),
    12: ("The principal contributions are: a unified inference path shared by the evaluation script, the "
         "command-line tool and the web interface; a reproducible protocol based on the public DIV2K dataset; "
         "and a benchmark showing that matching the training degradation to realistic inputs is essential, "
         "since models trained only on bicubic data perform worse than interpolation on degraded images."),
    15: ("SRCNN [1] is the baseline. It operates on a bicubically up-sampled input and learns a mapping to the "
         "HR image with a small three-layer convolutional network (9–5–5 kernels). In this project it "
         "processes RGB images, predicts a residual over the bicubic image, and contains 69.3 K parameters."),
    18: ("EDSR [2] is the primary, PSNR-oriented model. It uses residual blocks without batch normalization and "
         "a global skip connection. Up-sampling is performed inside the network by sub-pixel convolution "
         "(PixelShuffle) [4] rather than by bicubic pre-up-sampling, so all convolutions run at LR resolution. "
         "The EDSR-baseline configuration used here has 16 residual blocks with 64 channels and 1.52 M "
         "parameters at ×4."),
    19: "C. EDSR-real",
    20: ("Models trained on clean bicubic down-sampling generalize poorly to real photographs, which also contain "
         "blur, sensor noise and compression artefacts. Following the idea of Real-ESRGAN [9], EDSR-real uses the "
         "same EDSR architecture fine-tuned on randomly degraded inputs: isotropic or anisotropic Gaussian blur, "
         "down-sampling with a random kernel, Gaussian noise and JPEG compression (quality 30–95). It "
         "therefore learns to restore and up-sample in a single step."),
    21: ("No architectural novelty is claimed: SRCNN and EDSR are existing methods and are cited as such in "
         "the source code."),
    24: ("Fig. 1 shows the system organization. A shared core (sr/) provides the models, the dataset and "
         "degradation pipeline, the metrics and a tiled inference function. The training and evaluation "
         "scripts, the command-line tool and the web demo all import this core, so a model behaves identically "
         "in each context. Table I summarizes the technology stack."),
    # Table I
    31: "ML core", 32: "PyTorch 2.14 (CUDA); shared by training, evaluation, demo",
    33: "Training", 34: "L1 loss, Adam, cosine LR, bf16 mixed precision",
    35: "Data", 36: "DIV2K; memory-mapped NumPy crops, on-the-fly LR",
    37: "Interface", 38: "Gradio web demo, upscale.py CLI",
    39: "Hardware", 40: "NVIDIA RTX 4050 Laptop GPU (6 GB)",
    41: "Evaluation", 42: "PSNR/SSIM (scikit-image), Matplotlib",
    45: ("The pipeline loads DIV2K [8] HR images, converts them once to memory-mapped NumPy arrays and extracts "
         "random 192×192 HR patches, which are augmented with flips and 90° rotations. The matching "
         "48×48 LR patches are generated on the fly, either by bicubic down-sampling or by the real-world "
         "degradation model of Section II-C, so the ground truth is always available. Test images are cropped "
         "to a multiple of the scale factor (modcrop) before down-sampling."),
    46: ("DIV2K contains 800 training and 100 validation 2K images. Images 0001–0400 of the training set, "
         "the portion hosted by the mirror used, form the training data. Validation images 0801–0810 select "
         "the best checkpoint during training, and the remaining 90 images (0811–0900) form the held-out "
         "test set."),
    48: ("A single training script handles all models through command-line arguments (--model, --scale, "
         "--iters, --minutes, --degradation, --init, --resume). Training uses the L1 loss, the Adam optimizer "
         "with cosine learning-rate decay, batches of 16 patches and bfloat16 mixed precision. Validation "
         "PSNR-Y is computed every 1,000 iterations; each run saves best and last checkpoints and a CSV "
         "history from which loss and PSNR curves are plotted."),
    50: ("The evaluation script benchmarks every checkpoint against the bicubic baseline, reports PSNR-Y, "
         "SSIM-Y (Gaussian window) [7], the gain over bicubic and the parameter count, and saves side-by-side "
         "visual comparisons. PSNR is defined as"),
    52: ("where MAX is the peak pixel value and MSE the mean squared error, computed on the luminance (Y) "
         "channel with a 4-pixel border excluded, following common SR practice. In the real-world setting each "
         "test image is degraded with a fixed random seed, so every method receives an identical input."),
    53: "E. Inference Interfaces",
    54: ("Inference is exposed through two thin front ends over the shared module. Large images are processed "
         "in overlapping 256×256 LR tiles so that they fit in GPU memory. Table II lists the main scripts "
         "of the project."),
    56: "Project Scripts",
    57: "Script",
    59: "download_data.sh", 60: "Download DIV2K from mirrors",
    61: "prepare_data.py", 62: "Unzip; convert HR images to .npy",
    63: "train.py", 64: "Train or fine-tune a model",
    65: "evaluate.py", 66: "PSNR/SSIM tables and figures",
    67: "upscale.py", 68: "Upscale one image (CLI)",
    69: "app.py", 70: "Gradio web demo",
    71: "sr/models.py, sr/utils.py", 72: "Models, metrics, tiled inference",
    73: "sr/degrade.py", 74: "Real-world degradation model",
}

NEW.update({
    76: "F. Web Demonstration",
    77: ("The Gradio web application lets a user upload an image, choose a model, view the bicubic and "
         "super-resolved results side by side and download the output. Three modes are offered: up-scaling the "
         "uploaded image directly, and two simulation modes that first degrade the image (clean bicubic or "
         "real-world) and report the PSNR of both outputs against the original. The application runs locally "
         "on the GPU and loads all trained checkpoints at start-up."),
    79: ("Three checkpoints were trained at ×4 on an NVIDIA RTX 4050 Laptop GPU (6 GB): SRCNN for 4,000 "
         "iterations (6.5 min) and EDSR for 30,000 iterations (39 min), both with an initial learning rate of "
         "2×10⁻⁴, and EDSR-real, initialized from EDSR and fine-tuned for 12,000 iterations "
         "(22 min) with a learning rate of 1×10⁻⁴. The software environment was Python 3.13 "
         "with torch 2.14.1+cu126. All models were evaluated on the 90-image held-out test split under two "
         "settings, clean bicubic down-sampling and the real-world degradation model, with bicubic "
         "interpolation as the reference baseline. Checkpoints store their architecture, scale and degradation "
         "type so that they reload correctly at inference time."),
    80: ("Each component was verified by end-to-end tests: output geometry of every model at ×4, short "
         "training runs, the evaluation script, the command-line tool and the inference function of the web "
         "interface."),
    82: ("Tables III and IV report the measured results; Fig. 2 compares PSNR-Y across models in both "
         "settings. Timing values are per-image inference times for a 510×339 input (2040×1356 output) "
         "on the GPU; bicubic interpolation runs on the CPU."),
    84: "Test-Set Results, Clean Bicubic ×4",
    122: "Test-Set Results, Real-World Degradation ×4",
})

def table(start, tab):
    """Header at start..start+6, then 4 rows of 7 cells."""
    hdr = ["Model", "PSNR-Y", "SSIM-Y", "ΔPSNR", "ΔSSIM", "Params", "ms/img"]
    for k, h in enumerate(hdr):
        NEW[start + k] = h
    names = [("Bicubic", "Bicubic", "—"), ("SRCNN", "SRCNN", "69.3 K"),
             ("EDSR", "EDSR", "1.52 M"), ("EDSR-REAL", "EDSR-real", "1.52 M")]
    for r, (key, label, params) in enumerate(names):
        base = start + 7 + 7 * r
        dp = P(tab, key) - P(tab, "Bicubic")
        ds = S(tab, key) - S(tab, "Bicubic")
        cells = [label, f"{P(tab, key):.3f}", f"{S(tab, key):.4f}",
                 "—" if key == "Bicubic" else d(dp),
                 "—" if key == "Bicubic" else d4(ds),
                 params, f1(ms[key])]
        for k, c in enumerate(cells):
            NEW[base + k] = c

table(85, clean)
table(123, real)

NEW.update({
    160: "Fig. 2. PSNR-Y of each model on the test split, for clean and real-world ×4 inputs.",
    161: ("Three observations follow from the measurements. First, on clean inputs both learned models "
          f"outperform bicubic interpolation: SRCNN gains {g_srcnn:.2f} dB and EDSR {g_edsr:.2f} dB in PSNR-Y, "
          f"with SSIM-Y rising from {S(clean,'Bicubic'):.4f} to {S(clean,'EDSR'):.4f}. Second, on realistic "
          "inputs the ranking reverses for models trained only on bicubic data: SRCNN and EDSR fall below the "
          f"bicubic baseline ({d(r_srcnn)} and {d(r_edsr)} dB), because they treat noise and compression "
          "artefacts as detail and amplify them. EDSR-real, fine-tuned on the degradation model, gains "
          f"{r_real:.2f} dB over bicubic and {r_real_edsr:.2f} dB over EDSR, at a cost of {c_real_cost:.2f} dB on "
          "clean inputs. Third, inference cost depends on the resolution at which a network operates as much "
          f"as on its depth: SRCNN, which convolves at HR resolution, needs {f1(ms['SRCNN'])} ms per image, close "
          f"to the {f1(ms['EDSR'])} ms of the much larger EDSR, which works at LR resolution; bicubic "
          f"interpolation takes {f1(ms['Bicubic'])} ms. All models remain interactive on a laptop GPU."),
    162: ("These results show that the training degradation matters as much as the architecture: the same "
          "1.52 M-parameter network is the best model in one setting and worse than interpolation in the "
          "other. Visual inspection confirms the trend: EDSR produces sharper edges and textures than SRCNN on "
          "clean inputs, while EDSR-real produces clean but somewhat smoother outputs on degraded inputs, a "
          "known consequence of optimizing a pixel-wise L1 loss."),
    164: ("The reported numbers are measured but not research-grade. The principal limitations are the short "
          "training schedules, the use of 400 of the 800 DIV2K training images, the single scale factor (×4) "
          "and the synthetic nature of the real-world degradation model. Natural next steps are to train longer "
          "on the full DIV2K and Flickr2K sets, to add ×2 and ×3 models, which the code already "
          "supports, and to evaluate on the standard Set5, Set14, BSD100 and Urban100 benchmarks. Adding a "
          "perceptual [5] and adversarial [3] loss, as in SRGAN and Real-ESRGAN [9], and reporting LPIPS [6] "
          "would address the smoothness of the EDSR-real outputs."),
    166: ("This paper presented a complete, executable image super-resolution system that unifies two "
          "architectures, a real-world degradation model, a training and evaluation pipeline, a command-line "
          f"tool and a web interface under a single shared code path. On clean ×4 inputs EDSR improves on "
          f"bicubic interpolation by {g_edsr:.2f} dB PSNR-Y. On realistic degraded inputs, only the model "
          f"fine-tuned on matching degradations, EDSR-real, outperforms interpolation, by {r_real:.2f} dB. The "
          "system provides a transparent, reproducible foundation for longer training runs and perceptual-loss "
          "models."),
})

REF9 = ("X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-ESRGAN: Training real-world blind super-resolution "
        "with pure synthetic data,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW), 2021, "
        "pp. 1905–1914.")

# ---------------- apply ----------------
x = open(DOC, encoding="utf8").read()
head, body = x.split("<w:body>", 1)
spans = [m.span() for m in re.finditer(r"<w:p[ >].*?</w:p>", body, re.S)]
T = re.compile(r"<w:t(?: [^>]*)?>([^<]*)</w:t>")

def old_text(p):
    return "".join(T.findall(p))

pieces, last, log = [], 0, []
for i, (a, b) in enumerate(spans):
    p = body[a:b]
    if i in NEW:
        ts = list(T.finditer(p))
        assert ts, f"paragraph {i} has no text"
        m = ts[-1]  # replace the text of the last run (keeps 'Abstract—' / 'Index Terms—' prefixes)
        new = f'<w:t xml:space="preserve">{escape(NEW[i])}</w:t>'
        before = old_text(p)
        p = p[:m.start()] + new + p[m.end():]
        log.append((i, before[:60], NEW[i][:60]))
    if i == 175:  # append reference [9] after [8], cloning its formatting
        ref = p.replace("[8]", "[9]", 1)
        ts = list(T.finditer(ref))
        m = ts[-1]
        ref = ref[:m.start()] + f'<w:t xml:space="preserve">{escape(REF9)}</w:t>' + ref[m.end():]
        ref = re.sub(r'w14:paraId="[0-9A-F]{8}"', 'w14:paraId="3C9A1E57"', ref)
        p = p + ref
    pieces.append(body[last:a] + p)
    last = b
pieces.append(body[last:])
open(DOC, "w", encoding="utf8").write(head + "<w:body>" + "".join(pieces))

for i, o, n in log:
    print(f"{i:4d} | {o!r:64} -> {n!r}")
print(len(log), "paragraphs updated, reference [9] added")
