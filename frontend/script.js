// ===== Settings =====
// While the backend is not built yet, DEMO_MODE = true shows a fake result.
// When the backend is ready, change it to false.
const DEMO_MODE = true;
const API_URL = "http://127.0.0.1:5000/predict";
const MAX_SIZE_MB = 10;

// ===== Page elements =====
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewArea = document.getElementById("preview-area");
const previewImg = document.getElementById("preview-img");
const fileNameText = document.getElementById("file-name");
const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");
const loader = document.getElementById("loader");
const errorBox = document.getElementById("error-box");
const resultBox = document.getElementById("result");
const resultView = document.getElementById("result-view");
const resultFull = document.getElementById("result-full");
const confidenceText = document.getElementById("confidence-text");
const confidenceBar = document.getElementById("confidence-bar");

let selectedFile = null;

// ===== Choosing a file =====
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

// Drag and drop
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    hideError();
    resultBox.classList.add("hidden");

    const allowed = ["image/jpeg", "image/png"];
    if (!allowed.includes(file.type)) {
        showError("Please choose a JPG or PNG image.");
        return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        showError("The image is larger than " + MAX_SIZE_MB + " MB.");
        return;
    }

    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    fileNameText.textContent = file.name;
    previewArea.classList.remove("hidden");
    analyzeBtn.disabled = false;
    resetBtn.classList.remove("hidden");
}

// ===== Analyze =====
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    hideError();
    resultBox.classList.add("hidden");
    loader.classList.remove("hidden");
    analyzeBtn.disabled = true;

    try {
        const data = DEMO_MODE ? await fakePredict() : await realPredict();
        showResult(data);
    } catch (err) {
        showError(
            "Could not get a prediction. Is the backend running? (" + err.message + ")"
        );
    } finally {
        loader.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
});

// Sends the image to the backend
async function realPredict() {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error("Server error " + response.status);
    }
    return await response.json(); // expected: { "view": "PA", "confidence": 0.97 }
}

// Fake answer for testing the page without a backend
function fakePredict() {
    return new Promise((resolve) => {
        setTimeout(() => {
            const view = Math.random() > 0.5 ? "PA" : "AP";
            const confidence = 0.8 + Math.random() * 0.19;
            resolve({ view: view, confidence: confidence });
        }, 1500);
    });
}

// ===== Showing results =====
function showResult(data) {
    const percent = (data.confidence * 100).toFixed(1);

    resultView.textContent = data.view;
    resultFull.textContent =
        data.view === "PA" ? "Posterior-Anterior view" : "Anterior-Posterior view";
    confidenceText.textContent = percent + "%";
    confidenceBar.style.width = "0%";
    resultBox.classList.remove("hidden");

    // small delay so the bar animates
    setTimeout(() => {
        confidenceBar.style.width = percent + "%";
    }, 50);
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    errorBox.classList.add("hidden");
}

// ===== Clear =====
resetBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewArea.classList.add("hidden");
    resultBox.classList.add("hidden");
    resetBtn.classList.add("hidden");
    analyzeBtn.disabled = true;
    hideError();
});