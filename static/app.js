document.addEventListener("DOMContentLoaded", () => {
    // API base URL configuration (uses full localhost path if loaded via file:/// or other ports)
    const isLocalFileOrDev = window.location.protocol === "file:" || 
                             (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1" && window.location.hostname !== "0.0.0.0") ||
                             (window.location.port !== "8000" && window.location.port !== "");
    const API_BASE = isLocalFileOrDev ? "http://127.0.0.1:8000" : "";

    // DOM Elements
    const chatForm = document.getElementById("chat-submit-form");
    const promptTextarea = document.getElementById("prompt-textarea");
    const imageFileInput = document.getElementById("image-file-input");
    const dragUploadPanel = document.getElementById("drag-upload-panel");
    const uploadPromptText = document.getElementById("upload-prompt-text");
    const uploadPreviewContainer = document.getElementById("upload-preview-container");
    const uploadPreviewImg = document.getElementById("upload-preview-img");
    const previewFileName = document.getElementById("preview-file-name");
    const previewFileSize = document.getElementById("preview-file-size");
    const btnRemoveImage = document.getElementById("btn-remove-image");
    const chatDialogArea = document.getElementById("chat-dialog-area");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const pipelineStepper = document.getElementById("pipeline-stepper-widget");
    
    // Status Badges
    const statusVgg = document.getElementById("status-vgg16");
    const statusBlip = document.getElementById("status-blip");
    const statusClip = document.getElementById("status-clip");
    const statusFaiss = document.getElementById("status-faiss");
    const statusLlama = document.getElementById("status-llama2");
    const networkDot = document.getElementById("network-dot");
    const networkText = document.getElementById("network-text");

    // Diagnostic Elements
    const visionPlaceholder = document.getElementById("vision-placeholder");
    const visionAnalysisContent = document.getElementById("vision-analysis-content");
    const dashScanImg = document.getElementById("dash-scan-img");
    const dashVggClass = document.getElementById("dash-vgg-class");
    const dashVggConfidenceFill = document.getElementById("dash-vgg-confidence-fill");
    const dashVggConfidenceText = document.getElementById("dash-vgg-confidence-text");
    const dashBlipCaption = document.getElementById("dash-blip-caption");
    const dashClipScore = document.getElementById("dash-clip-score");
    const dashClipFill = document.getElementById("dash-clip-fill");
    
    const ragPlaceholder = document.getElementById("rag-placeholder");
    const ragAnalysisContent = document.getElementById("rag-analysis-content");
    const chunkTextSimilar = document.getElementById("chunk-text-similar");
    const chunkImageSimilar = document.getElementById("chunk-image-similar");
    const chunkImageTextSimilar = document.getElementById("chunk-imagetext-similar");

    // State Variables
    let selectedImageFile = null;
    let stepperInterval = null;

    // 1. Check Model activation status on boot
    async function checkModelStatus() {
        try {
            const res = await fetch(`${API_BASE}/api/status`);
            const data = await res.json();
            
            updateStatusBadge(statusVgg, data.vgg16);
            updateStatusBadge(statusBlip, data.blip);
            updateStatusBadge(statusClip, data.clip);
            updateStatusBadge(statusFaiss, data.faiss);
            updateStatusBadge(statusLlama, data.llama2);

            // Check if running in simulation mode
            const isDemo = Object.values(data).some(val => val.includes("Mock") || val.includes("Simulation"));
            if (isDemo) {
                networkDot.className = "status-dot pulsing";
                networkText.innerText = "Simulation Core Active";
            } else {
                networkDot.className = "status-dot loaded";
                networkText.innerText = "Offline Hardware Active";
            }
        } catch (err) {
            console.error("Failed to read server status API", err);
            [statusVgg, statusBlip, statusClip, statusFaiss, statusLlama].forEach(el => {
                el.innerText = "Error";
                el.className = "matrix-val badge-warning";
            });
        }
    }

    function updateStatusBadge(element, statusText) {
        element.innerText = statusText.includes("Active") ? "Active" : "Simulation";
        if (statusText.includes("Active")) {
            element.className = "matrix-val badge-active";
        } else {
            element.className = "matrix-val badge-warning";
        }
    }

    // Call status immediately
    checkModelStatus();

    // 2. Drag & Drop File Upload Interactions
    dragUploadPanel.addEventListener("click", () => {
        if (!selectedImageFile) {
            imageFileInput.click();
        }
    });

    dragUploadPanel.addEventListener("dragover", (e) => {
        e.preventDefault();
        dragUploadPanel.classList.add("dragging");
    });

    ["dragleave", "dragend"].forEach(type => {
        dragUploadPanel.addEventListener(type, () => {
            dragUploadPanel.classList.remove("dragging");
        });
    });

    dragUploadPanel.addEventListener("drop", (e) => {
        e.preventDefault();
        dragUploadPanel.classList.remove("dragging");
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            handleFileSelect(file);
        }
    });

    imageFileInput.addEventListener("change", () => {
        if (imageFileInput.files.length > 0) {
            const file = imageFileInput.files[0];
            handleFileSelect(file);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Format unsupported. Please select a valid image file.");
            return;
        }
        selectedImageFile = file;
        
        // Render File Details
        previewFileName.innerText = file.name;
        previewFileSize.innerText = formatBytes(file.size);
        
        // Show Image Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreviewImg.src = e.target.result;
            uploadPromptText.classList.add("hidden");
            uploadPreviewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    btnRemoveImage.addEventListener("click", (e) => {
        e.stopPropagation(); // Avoid triggering parent click
        clearImageSelection();
    });

    function clearImageSelection() {
        selectedImageFile = null;
        imageFileInput.value = "";
        uploadPreviewImg.src = "";
        uploadPreviewContainer.classList.add("hidden");
        uploadPromptText.classList.remove("hidden");
    }

    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // 3. Textarea Auto-growth and Submit bindings
    promptTextarea.addEventListener("input", () => {
        promptTextarea.style.height = "auto";
        promptTextarea.style.height = promptTextarea.scrollHeight + "px";
    });

    promptTextarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // 4. Reset Conversation logs
    clearChatBtn.addEventListener("click", () => {
        if (confirm("Reset current conversation logs?")) {
            chatDialogArea.innerHTML = `
                <div class="message system-msg">
                    <div class="message-avatar">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                            <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                        </svg>
                    </div>
                    <div class="message-bubble">
                        <p><strong>Aegis Security System Initialized.</strong> Your chatbot session is fully isolated and secure. Upload chest X-rays or diagnostic images and query the RAG medical document library without exposing data online.</p>
                    </div>
                </div>
            `;
            // Reset Dashboard
            visionPlaceholder.classList.remove("hidden");
            visionAnalysisContent.classList.add("hidden");
            ragPlaceholder.classList.remove("hidden");
            ragAnalysisContent.classList.add("hidden");
            pipelineStepper.classList.remove("active");
            clearImageSelection();
        }
    });

    // 5. Submit chat logic
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const queryText = promptTextarea.value.trim();
        if (!queryText && !selectedImageFile) return;

        // Save refs and clear UI immediately
        const tempText = queryText;
        const tempImage = selectedImageFile;
        
        promptTextarea.value = "";
        promptTextarea.style.height = "auto";
        clearImageSelection();

        // 5a. Append User Message
        appendUserMessage(tempText, tempImage);
        
        // 5b. Append Bot Loading state
        const botLoadingId = appendBotLoading();

        // 5c. Start Pipeline Stepper Animation
        startPipelineStepper(tempImage !== null);

        // 5d. Dispatch Request
        const formData = new FormData();
        formData.append("text", tempText);
        if (tempImage) {
            formData.append("image", tempImage);
        }

        try {
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server returned code ${response.status}`);
            }

            const data = await response.json();
            
            // Remove loader and display chatbot output
            removeBotLoading(botLoadingId);
            appendBotMessage(data.output);
            
            // Finalize stepper animation
            completePipelineStepper();

            // 5e. Update Dashboard Panels
            updateDashboardMetrics(data, tempImage !== null);

        } catch (err) {
            console.error("Chat dispatch error", err);
            removeBotLoading(botLoadingId);
            appendBotMessage(`[System Red Alert - Inference Error]: Failed to query local LLM nodes. Details: ${err.message}. Ensure requirements are installed and backend service is running.`);
            resetPipelineStepper();
        }
    });

    // Chat Append Utilities
    function appendUserMessage(text, imgFile) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message user-msg";
        
        let imgHtml = "";
        if (imgFile) {
            const objectUrl = URL.createObjectURL(imgFile);
            imgHtml = `<img src="${objectUrl}" alt="uploaded scan" onload="window.scrollToBottom()">`;
        }

        msgDiv.innerHTML = `
            <div class="message-avatar">U</div>
            <div class="message-bubble">
                <p>${escapeHtml(text)}</p>
                ${imgHtml}
            </div>
        `;
        chatDialogArea.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendBotLoading() {
        const id = "loading-" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.className = "message bot-msg";
        msgDiv.id = id;
        msgDiv.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-bubble">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        chatDialogArea.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function removeBotLoading(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendBotMessage(text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message bot-msg";
        
        // Simple linebreaks to HTML paragraphs
        const formattedText = text.split("\n").map(para => `<p>${escapeHtml(para)}</p>`).join("");

        msgDiv.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-bubble">
                ${formattedText}
            </div>
        `;
        chatDialogArea.appendChild(msgDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatDialogArea.scrollTop = chatDialogArea.scrollHeight;
    }
    
    // Bind to window for image loads
    window.scrollToBottom = scrollToBottom;

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Pipeline Stepper Progress controller
    function startPipelineStepper(hasImage) {
        pipelineStepper.classList.add("active");
        
        // Reset steps
        const steps = ["step-vgg", "step-blip", "step-clip", "step-faiss", "step-llama"];
        steps.forEach(id => {
            document.getElementById(id).className = "step";
        });

        let currentIdx = 0;
        
        // Heuristic stepper updates to represent model pipeline flow
        const runStepSequence = () => {
            // If query doesn't have an image, skip image steps (VGG, BLIP, CLIP)
            if (!hasImage && currentIdx < 3) {
                // Instantly complete visual steps 1, 2, 3 as they are skipped
                for (let i = 0; i < 3; i++) {
                    const stepEl = document.getElementById(steps[i]);
                    stepEl.className = "step completed";
                }
                currentIdx = 3;
            }

            // Reset previously active step to completed
            if (currentIdx > 0) {
                document.getElementById(steps[currentIdx - 1]).className = "step completed";
            }
            
            if (currentIdx < steps.length) {
                document.getElementById(steps[currentIdx]).className = "step active";
                currentIdx++;
            } else {
                clearInterval(stepperInterval);
            }
        };

        // Run immediately
        runStepSequence();
        // Progress step status every 1.5 seconds
        clearInterval(stepperInterval);
        stepperInterval = setInterval(runStepSequence, 1500);
    }

    function completePipelineStepper() {
        clearInterval(stepperInterval);
        const steps = ["step-vgg", "step-blip", "step-clip", "step-faiss", "step-llama"];
        steps.forEach(id => {
            document.getElementById(id).className = "step completed";
        });
        
        // Keep stepper visible for 1.5 seconds then fade out smoothly
        setTimeout(() => {
            if (pipelineStepper.classList.contains("active")) {
                pipelineStepper.classList.remove("active");
            }
        }, 2000);
    }

    function resetPipelineStepper() {
        clearInterval(stepperInterval);
        pipelineStepper.classList.remove("active");
    }

    // 6. Update Dashboard Analytics Columns
    function updateDashboardMetrics(data, hasImage) {
        // Toggle Image Insight panels
        if (hasImage) {
            visionPlaceholder.classList.add("hidden");
            visionAnalysisContent.classList.remove("hidden");
            
            // Set previews
            dashScanImg.src = data.image_url ? (data.image_url.startsWith("http") ? data.image_url : (API_BASE + data.image_url)) : "";
            
            // VGG16 updates
            dashVggClass.innerText = data.classification;
            const vggConfPct = (data.vgg_confidence * 100).toFixed(1);
            dashVggConfidenceText.innerText = vggConfPct + "%";
            dashVggConfidenceFill.style.width = vggConfPct + "%";

            // BLIP updates
            dashBlipCaption.innerText = `"${data.caption}"`;

            // CLIP similarity updates
            const clipVal = data.clip_score.toFixed(2);
            dashClipScore.innerText = clipVal;
            dashClipFill.style.width = (data.clip_score * 100).toFixed(0) + "%";
        } else {
            visionPlaceholder.classList.remove("hidden");
            visionAnalysisContent.classList.add("hidden");
        }

        // FAISS context updates
        if (data.text_similar && data.text_similar.length > 0) {
            ragPlaceholder.classList.add("hidden");
            ragAnalysisContent.classList.remove("hidden");
            
            chunkTextSimilar.innerText = data.text_similar[0];
            
            if (hasImage && data.image_similar && data.image_similar.length > 0) {
                chunkImageSimilar.innerText = data.image_similar[0];
                chunkImageTextSimilar.innerText = data.image_text_similar[0];
                
                // Show items
                chunkImageSimilar.closest(".accordion-item").classList.remove("hidden");
                chunkImageTextSimilar.closest(".accordion-item").classList.remove("hidden");
            } else {
                // Hide image-based document retrieve cards if not active
                chunkImageSimilar.closest(".accordion-item").classList.add("hidden");
                chunkImageTextSimilar.closest(".accordion-item").classList.add("hidden");
            }
        } else {
            ragPlaceholder.classList.remove("hidden");
            ragAnalysisContent.classList.add("hidden");
        }

        // IPFS block simulation trigger - randomized block indicators on query success
        triggerIPFSChunkBlink();
    }

    function triggerIPFSChunkBlink() {
        const dots = document.querySelectorAll("#chunk-dots-grid .chunk-dot");
        dots.forEach(dot => {
            // Briefly blink active items to represent dynamic data retrieval routing
            if (Math.random() > 0.4) {
                dot.style.opacity = "0.3";
                setTimeout(() => {
                    dot.style.opacity = "1";
                    // Toggle status randomly to show distributed replica updates
                    if (Math.random() > 0.8) {
                        dot.classList.toggle("active");
                    }
                }, Math.random() * 800 + 200);
            }
        });
    }

    // Accordion handler binding
    window.toggleAccordion = function(header) {
        const item = header.closest('.accordion-item');
        item.classList.toggle('open');
    };
});
