() => {
    // Singleton guard - ensures listeners and observers are bound strictly once
    if (window._pdf_app_initialized) return;
    window._pdf_app_initialized = true;

    // --- 1. FOCUS VIEWPORT GEOMETRIC TRANSFORM ---
    function applyFocusTransform(img, viewport, fx, fy, fw, fh) {
        if (!img || !viewport) return;

        function compute() {
            const vw = viewport.clientWidth;
            const vh = viewport.clientHeight;
            const iw = img.naturalWidth;
            const ih = img.naturalHeight;

            if (!vw || !vh || !iw || !ih) return;

            // Rendered display dimensions of full-width image (100% width)
            const renderedW = vw;
            const renderedH = vw * (ih / iw);

            // Normalized focus box dimensions and center point
            const safeFw = Math.max(fw, 0.20);
            const safeFh = Math.max(fh, 0.15);
            const cx = fx + fw / 2;
            const cy = fy + fh / 2;

            // Compute optimum scale to fit the focus area in the viewport
            const scaleX = 1 / safeFw;
            const scaleY = (vh / renderedH) / safeFh;
            const scale = Math.min(Math.max(Math.min(scaleX, scaleY), 1.25), 3.2);

            // Target focus center in rendered pixel coordinates
            const focusCenterX = cx * renderedW;
            const focusCenterY = cy * renderedH;

            // Viewport center
            const vCenterX = vw / 2;
            const vCenterY = vh / 2;

            // Shift focus center to viewport center
            let transX = (vCenterX - focusCenterX * scale);
            let transY = (vCenterY - focusCenterY * scale);

            // Keep bounds within reasonable view
            const maxTransX = 0;
            const minTransX = -(renderedW * scale - vw);
            const maxTransY = 0;
            const minTransY = -(renderedH * scale - vh);

            if (minTransX <= 0) {
                transX = Math.max(minTransX, Math.min(maxTransX, transX));
            }
            if (minTransY <= 0) {
                transY = Math.max(minTransY, Math.min(maxTransY, transY));
            }

            img.style.transformOrigin = '0 0';
            img.style.transform = `translate(${transX.toFixed(1)}px, ${transY.toFixed(1)}px) scale(${scale.toFixed(2)})`;
        }

        if (img.complete && img.naturalWidth > 0) {
            compute();
        } else {
            img.onload = () => {
                compute();
            };
        }
    }

    // --- 2. SELECT RESULT & UPDATE RIGHT PANEL ---
    window.selectPdfResult = function(idx) {
        const cards = document.querySelectorAll('#search-results .result-card');
        if (!cards || cards.length === 0) return;

        let targetCard = null;
        cards.forEach((c) => {
            const cIdx = parseInt(c.getAttribute('data-index'), 10);
            if (cIdx === idx) {
                c.classList.add('active-result-card');
                targetCard = c;
            } else {
                c.classList.remove('active-result-card');
            }
        });

        if (!targetCard) {
            targetCard = cards[0];
            targetCard.classList.add('active-result-card');
        }

        const selectedIdx = parseInt(targetCard.getAttribute('data-index') || '0', 10);
        const parentDetails = targetCard.closest('details');
        if (parentDetails && !parentDetails.open) {
            parentDetails.open = true;
        }

        const thumbnails = document.querySelectorAll('#preview-thumbnail-strip .preview-thumbnail');
        thumbnails.forEach((thumbnail) => {
            const thumbnailIdx = parseInt(thumbnail.getAttribute('data-result-index') || '-1', 10);
            const isSelected = thumbnailIdx === selectedIdx;
            const wasSelected = thumbnail.classList.contains('is-active');
            thumbnail.classList.toggle('is-active', isSelected);
            if (isSelected) {
                thumbnail.setAttribute('aria-current', 'true');
                if (!wasSelected) {
                    thumbnail.scrollIntoView({
                        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
                        block: 'nearest',
                        inline: 'nearest'
                    });
                }
            } else {
                thumbnail.removeAttribute('aria-current');
            }
        });

        const fullUrl = targetCard.getAttribute('data-full-url') || '';
        const fileName = targetCard.getAttribute('data-file-name') || 'Belge';
        const pageNum = targetCard.getAttribute('data-page-num') || '1';
        const totalPages = targetCard.getAttribute('data-total-pages') || '1';
        const fx = parseFloat(targetCard.getAttribute('data-focus-x')) || 0.0;
        const fy = parseFloat(targetCard.getAttribute('data-focus-y')) || 0.0;
        const fw = parseFloat(targetCard.getAttribute('data-focus-w')) || 1.0;
        const fh = parseFloat(targetCard.getAttribute('data-focus-h')) || 1.0;

        // Update preview header elements
        const nameEl = document.getElementById('preview-file-name');
        if (nameEl) nameEl.textContent = fileName;

        const badgeEl = document.getElementById('preview-page-badge');
        if (badgeEl) badgeEl.textContent = `Sayfa ${pageNum} / ${totalPages}`;

        // Update viewport image & transform
        const img = document.getElementById('preview-viewport-img');
        const viewport = document.getElementById('preview-viewport');
        if (img && viewport && fullUrl) {
            if (img.getAttribute('src') !== fullUrl) {
                img.src = fullUrl;
            }
            applyFocusTransform(img, viewport, fx, fy, fw, fh);
        }
    };

    // --- 3. MODAL STATE & CONTROLLER ---
    let modalZoom = 1.0;
    let modalPanX = 0;
    let modalPanY = 0;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let initialPanX = 0;
    let initialPanY = 0;
    let lastActiveTrigger = null;

    function clampModalPan() {
        const viewport = document.getElementById('modal-viewport');
        const image = document.getElementById('modal-full-image');

        if (!viewport || !image) {
            modalPanX = 0;
            modalPanY = 0;
            return;
        }

        const viewportWidth = viewport.clientWidth;
        const viewportHeight = viewport.clientHeight;
        const scaledWidth = image.offsetWidth * modalZoom;
        const scaledHeight = image.offsetHeight * modalZoom;

        const maxPanX = Math.max(0, (scaledWidth - viewportWidth) / 2);
        const maxPanY = Math.max(0, (scaledHeight - viewportHeight) / 2);

        modalPanX = Math.max(-maxPanX, Math.min(maxPanX, modalPanX));
        modalPanY = Math.max(-maxPanY, Math.min(maxPanY, modalPanY));
    }

    function updateModalTransform() {
        const modalImg = document.getElementById('modal-full-image');
        const zoomLabel = document.getElementById('modal-zoom-label');
        if (modalImg) {
            modalImg.style.transform = `translate(${modalPanX.toFixed(1)}px, ${modalPanY.toFixed(1)}px) scale(${modalZoom.toFixed(2)})`;
        }
        if (zoomLabel) {
            zoomLabel.textContent = `${Math.round(modalZoom * 100)}%`;
        }
    }

    window.openPdfModal = function(triggerEl) {
        const activeCard = document.querySelector('#search-results .active-result-card') ||
                           document.querySelector('#search-results .result-card');
        if (!activeCard) return;

        const fullUrl = activeCard.getAttribute('data-full-url');
        if (!fullUrl) return;

        const fileName = activeCard.getAttribute('data-file-name') || 'Belge';
        const pageNum = activeCard.getAttribute('data-page-num') || '1';
        const totalPages = activeCard.getAttribute('data-total-pages') || '1';

        lastActiveTrigger = triggerEl || document.activeElement;

        const modal = document.getElementById('pdf-viewer-modal');
        const modalImg = document.getElementById('modal-full-image');
        const modalFilename = document.getElementById('modal-filename');
        const modalPagebadge = document.getElementById('modal-pagebadge');

        if (modalFilename) modalFilename.textContent = fileName;
        if (modalPagebadge) modalPagebadge.textContent = `Sayfa ${pageNum} / ${totalPages}`;

        // Reset zoom & pan
        modalZoom = 1.0;
        modalPanX = 0;
        modalPanY = 0;
        updateModalTransform();

        if (modalImg) {
            if (modalImg.getAttribute('src') !== fullUrl) {
                modalImg.src = fullUrl;
            }
            if (modalImg.complete && modalImg.naturalWidth > 0) {
                clampModalPan();
                updateModalTransform();
            } else {
                modalImg.onload = () => {
                    clampModalPan();
                    updateModalTransform();
                };
            }
        }

        if (modal) {
            modal.classList.add('modal-active');
            modal.setAttribute('aria-hidden', 'false');
            document.body.classList.add('modal-open');

            const closeBtn = document.getElementById('modal-close');
            if (closeBtn) closeBtn.focus();
        }
    };

    window.closePdfModal = function() {
        const modal = document.getElementById('pdf-viewer-modal');
        if (modal) {
            modal.classList.remove('modal-active');
            modal.setAttribute('aria-hidden', 'true');
        }
        document.body.classList.remove('modal-open');

        modalZoom = 1.0;
        modalPanX = 0;
        modalPanY = 0;
        updateModalTransform();

        if (lastActiveTrigger && typeof lastActiveTrigger.focus === 'function') {
            try {
                lastActiveTrigger.focus();
            } catch (err) {}
        }
    };

    // --- 4. BIND MODAL LISTENERS (ONCE) ---
    function initModal() {
        const modal = document.getElementById('pdf-viewer-modal');
        const modalBackdrop = document.getElementById('modal-backdrop');
        const modalClose = document.getElementById('modal-close');
        const modalZoomIn = document.getElementById('modal-zoom-in');
        const modalZoomOut = document.getElementById('modal-zoom-out');
        const modalZoomReset = document.getElementById('modal-zoom-reset');
        const modalCanvas = document.getElementById('modal-canvas');

        if (!modal) return;

        if (modalClose) {
            modalClose.addEventListener('click', (e) => {
                e.stopPropagation();
                window.closePdfModal();
            });
        }

        if (modalBackdrop) {
            modalBackdrop.addEventListener('click', (e) => {
                e.stopPropagation();
                window.closePdfModal();
            });
        }

        if (modalZoomIn) {
            modalZoomIn.addEventListener('click', (e) => {
                e.stopPropagation();
                modalZoom = Math.min(modalZoom + 0.25, 3.0);
                clampModalPan();
                updateModalTransform();
            });
        }

        if (modalZoomOut) {
            modalZoomOut.addEventListener('click', (e) => {
                e.stopPropagation();
                modalZoom = Math.max(modalZoom - 0.25, 0.5);
                clampModalPan();
                updateModalTransform();
            });
        }

        if (modalZoomReset) {
            modalZoomReset.addEventListener('click', (e) => {
                e.stopPropagation();
                modalZoom = 1.0;
                modalPanX = 0;
                modalPanY = 0;
                updateModalTransform();
            });
        }

        // Pointer Capture Pan/Drag with Geometric Clamping
        if (modalCanvas) {
            modalCanvas.addEventListener('pointerdown', (e) => {
                isDragging = true;
                modalCanvas.classList.add('is-dragging');
                dragStartX = e.clientX;
                dragStartY = e.clientY;
                initialPanX = modalPanX;
                initialPanY = modalPanY;
                try {
                    modalCanvas.setPointerCapture(e.pointerId);
                } catch (err) {}
            });

            modalCanvas.addEventListener('pointermove', (e) => {
                if (!isDragging) return;
                const dx = e.clientX - dragStartX;
                const dy = e.clientY - dragStartY;

                modalPanX = initialPanX + dx;
                modalPanY = initialPanY + dy;
                clampModalPan();
                updateModalTransform();
            });

            const stopDrag = (e) => {
                if (isDragging) {
                    isDragging = false;
                    modalCanvas.classList.remove('is-dragging');
                    try {
                        modalCanvas.releasePointerCapture(e.pointerId);
                    } catch (err) {}
                }
            };

            modalCanvas.addEventListener('pointerup', stopDrag);
            modalCanvas.addEventListener('pointercancel', stopDrag);
        }

        // Keyboard navigation & focus trap
        document.addEventListener('keydown', (e) => {
            const isModalActive = modal.classList.contains('modal-active');
            if (!isModalActive) return;

            if (e.key === 'Escape') {
                e.preventDefault();
                window.closePdfModal();
                return;
            }

            if (e.key === 'Tab') {
                const focusable = modal.querySelectorAll('button:not([disabled])');
                if (focusable.length === 0) return;

                const firstEl = focusable[0];
                const lastEl = focusable[focusable.length - 1];

                if (e.shiftKey) {
                    if (document.activeElement === firstEl) {
                        e.preventDefault();
                        lastEl.focus();
                    }
                } else {
                    if (document.activeElement === lastEl) {
                        e.preventDefault();
                        firstEl.focus();
                    }
                }
            }
        });
    }

    // --- 5. EVENT DELEGATION & RESIZE OBSERVER (SINGLETON) ---
    document.addEventListener('click', function(e) {
        const thumbnail = e.target.closest('#preview-thumbnail-strip .preview-thumbnail');
        if (thumbnail) {
            e.preventDefault();
            e.stopPropagation();
            const thumbnailIdx = parseInt(thumbnail.getAttribute('data-result-index') || '-1', 10);
            if (!isNaN(thumbnailIdx) && thumbnailIdx >= 0) {
                window.selectPdfResult(thumbnailIdx);
            }
            return;
        }

        // Result card click (delegated exclusively under #search-results)
        const card = e.target.closest('#search-results .result-card');
        if (card) {
            const idxAttr = card.getAttribute('data-index');
            if (idxAttr !== null) {
                const idx = parseInt(idxAttr, 10);
                if (!isNaN(idx)) {
                    window.selectPdfResult(idx);
                }
            }
        }
    });

    // Window resize handler to recalculate focused preview transform and clamp open modal
    window.addEventListener('resize', () => {
        const activeCard = document.querySelector('#search-results .active-result-card') ||
                           document.querySelector('#search-results .result-card');
        if (activeCard) {
            const idx = parseInt(activeCard.getAttribute('data-index') || '0', 10);
            window.selectPdfResult(idx);
        }
        const modal = document.getElementById('pdf-viewer-modal');
        if (modal && modal.classList.contains('modal-active')) {
            clampModalPan();
            updateModalTransform();
        }
    });

    // --- 6. SINGLETON MUTATION OBSERVER ON SEARCH RESULTS ---
    function initSearchObserver() {
        const resultsEl = document.getElementById('search-results');
        const countBadge = document.getElementById('results-count');
        if (!resultsEl) return;

        function updateResultCount() {
            const cards = resultsEl.querySelectorAll('.result-card');
            const count = cards.length;
            if (countBadge) {
                countBadge.textContent = `${count} SONUÇ`;
            }
        }

        const observer = new MutationObserver(() => {
            updateResultCount();
            const firstCard = resultsEl.querySelector('.result-card');
            if (firstCard) {
                // Ensure first card is selected and focused preview is applied
                setTimeout(() => {
                    window.selectPdfResult(0);
                }, 50);
            }
        });

        observer.observe(resultsEl, { childList: true, subtree: true });
        updateResultCount();
    }

    // --- 7. UNIFIED ACCORDION HEADER & CONTENT ENHANCEMENT ---
    function initAccordionHeader() {
        function enhanceHeader() {
            const acc = document.getElementById('library-accordion');
            if (!acc) return false;
            const labelWrap = acc.querySelector('.label-wrap');
            if (!labelWrap) return false;

            // 1. Check if brandContainer already exists
            let brandContainer = labelWrap.querySelector('.accordion-brand-container');
            if (!brandContainer) {
                brandContainer = document.createElement('div');
                brandContainer.className = 'accordion-brand-container';
                brandContainer.innerHTML = `
                    <div class="accordion-folder-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0F172A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>
                        </svg>
                    </div>
                    <div class="accordion-title-group">
                        <span class="accordion-main-title">Belge Kütüphanesi</span>
                        <span class="accordion-app-subtitle">Akıllı PDF Arama & Görsel Keşif Platformu</span>
                    </div>
                `;
                labelWrap.insertBefore(brandContainer, labelWrap.firstChild);
            }

            // 2. Hide only the default title text span, preserve .icon / svg chevron
            const spans = labelWrap.querySelectorAll('span');
            spans.forEach(s => {
                if (
                    !s.closest('.accordion-brand-container') &&
                    !s.closest('.accordion-badges-group') &&
                    !s.classList.contains('icon') &&
                    !s.querySelector('svg')
                ) {
                    s.style.display = 'none';
                }
            });

            // 3. Ensure the chevron .icon is visible and positioned at the far right
            let iconEl = labelWrap.querySelector('.icon');
            if (iconEl) {
                iconEl.style.display = 'inline-flex';
                iconEl.style.visibility = 'visible';
                iconEl.style.opacity = '1';
                if (!iconEl.querySelector('svg')) {
                    iconEl.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>`;
                }
            } else {
                iconEl = document.createElement('span');
                iconEl.className = 'icon';
                iconEl.style.display = 'inline-flex';
                iconEl.style.visibility = 'visible';
                iconEl.style.opacity = '1';
                iconEl.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>`;
                labelWrap.appendChild(iconEl);
            }

            // 4. Inject or update minimalist badges group
            let badgesGroup = labelWrap.querySelector('.accordion-badges-group');
            if (!badgesGroup) {
                badgesGroup = document.createElement('div');
                badgesGroup.className = 'accordion-badges-group';

                const metaEl = document.getElementById('library-stats-meta');
                const docCount = metaEl ? (metaEl.getAttribute('data-doc-count') || '0') : '0';
                const health = metaEl ? (metaEl.getAttribute('data-health') || 'Sağlıklı') : 'Sağlıklı';
                const timeStr = metaEl ? (metaEl.getAttribute('data-time') || 'Az önce') : 'Az önce';

                badgesGroup.innerHTML = `
                    <span class="accordion-badge-pill accordion-badge-doc-count">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                        <span class="badge-count-text">${docCount} Belge</span>
                    </span>
                    <span class="accordion-badge-pill accordion-badge-health">
                        <span class="badge-dot-green">●</span>
                        <span class="badge-health-text">${health}</span>
                    </span>
                    <span class="accordion-badge-pill accordion-badge-time">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                        <span class="badge-time-text">Son Güncelleme: ${timeStr}</span>
                    </span>
                `;
                labelWrap.insertBefore(badgesGroup, iconEl);
            }

            if (!labelWrap._accordion_click_bound) {
                labelWrap._accordion_click_bound = true;
                labelWrap.addEventListener('click', () => {
                    setTimeout(enhanceContent, 50);
                    setTimeout(enhanceContent, 200);
                    setTimeout(enhanceContent, 500);
                });
            }

            return true;
        }

        function enhanceContent() {
            let allDone = true;

            // 5. Enhance #pdf-upload-zone with custom visual overlay
            const uploadZone = document.getElementById('pdf-upload-zone');
            if (uploadZone) {
                if (!uploadZone.querySelector('.custom-drop-overlay')) {
                    const overlay = document.createElement('div');
                    overlay.className = 'custom-drop-overlay';
                    overlay.innerHTML = `
                        <div class="upload-icon-circle">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path>
                                <path d="M12 12v9"></path>
                                <path d="m16 16-4-4-4 4"></path>
                            </svg>
                        </div>
                        <div class="upload-title">PDF Dosyalarını Buraya Sürükleyin</div>
                        <div class="upload-subtitle">veya bilgisayarınızdan seçmek için tıklayın</div>
                        <div class="upload-meta-pills">
                            <span class="upload-pill-badge">Maks. 50 MB • Yalnızca PDF</span>
                            <span class="upload-pill-badge upload-pill-action">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="12" y1="18" x2="12" y2="12"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg>
                                Dosya Seç
                            </span>
                        </div>
                    `;
                    uploadZone.appendChild(overlay);

                    ['dragenter', 'dragover'].forEach(eventName => {
                        uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
                    });
                    ['dragleave', 'drop'].forEach(eventName => {
                        uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
                    });
                }
            } else {
                allDone = false;
            }

            // 6. Enhance #index-button with orange sync icon
            const idxBtn = document.getElementById('index-button');
            if (idxBtn) {
                if (!idxBtn.querySelector('.index-btn-icon')) {
                    const targetBtn = idxBtn.querySelector('button') || idxBtn;
                    const iconSpan = document.createElement('span');
                    iconSpan.className = 'index-btn-icon';
                    iconSpan.style.display = 'inline-flex';
                    iconSpan.style.alignItems = 'center';
                    iconSpan.style.marginRight = '8px';
                    iconSpan.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F97316" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 21h5v-5"/></svg>`;
                    targetBtn.insertBefore(iconSpan, targetBtn.firstChild);
                }
            } else {
                allDone = false;
            }

            // 7. Dynamic reactive badge updating from #index-status
            const statusBox = document.getElementById('index-status');
            if (statusBox && !statusBox._observer_bound) {
                statusBox._observer_bound = true;
                const updateBadgesFromStatus = () => {
                    const textarea = statusBox.querySelector('textarea') || statusBox.querySelector('input');
                    const text = textarea ? textarea.value : statusBox.innerText;
                    if (!text) return;

                    const countEl = document.querySelector('.badge-count-text');
                    const healthEl = document.querySelector('.badge-health-text');
                    const timeEl = document.querySelector('.badge-time-text');

                    const indexMatch = text.match(/Eklenen:\s*(\d+).*?Güncellenen:\s*(\d+).*?Atlanan:\s*(\d+)/i);
                    if (indexMatch) {
                        const added = parseInt(indexMatch[1], 10) || 0;
                        const updated = parseInt(indexMatch[2], 10) || 0;
                        const skipped = parseInt(indexMatch[3], 10) || 0;
                        const total = added + updated + skipped;
                        if (countEl && total > 0) {
                            countEl.textContent = `${total} Belge`;
                        }
                        if (healthEl) {
                            healthEl.textContent = 'Sağlıklı';
                        }
                        if (timeEl) {
                            timeEl.textContent = 'Son Güncelleme: Az önce';
                        }
                    }

                    const uploadMatch = text.match(/(\d+)\s*yeni PDF/i);
                    if (uploadMatch && countEl) {
                        const addedUploads = parseInt(uploadMatch[1], 10) || 0;
                        const currentCount = parseInt(countEl.textContent, 10) || 0;
                        countEl.textContent = `${currentCount + addedUploads} Belge`;
                    }
                };

                const obs = new MutationObserver(updateBadgesFromStatus);
                obs.observe(statusBox, { childList: true, subtree: true, characterData: true });
                const textarea = statusBox.querySelector('textarea') || statusBox.querySelector('input');
                if (textarea) {
                    textarea.addEventListener('input', updateBadgesFromStatus);
                    textarea.addEventListener('change', updateBadgesFromStatus);
                }
            } else if (!statusBox) {
                allDone = false;
            }

            return allDone;
        }

        // Run on load and poll briefly for full mount
        enhanceHeader();
        enhanceContent();

        let attempts = 0;
        const timer = setInterval(() => {
            attempts++;
            enhanceHeader();
            const done = enhanceContent();
            if ((done && attempts > 5) || attempts > 35) {
                clearInterval(timer);
            }
        }, 150);

        // Also observe library-accordion subtree
        const acc = document.getElementById('library-accordion');
        if (acc) {
            const accObserver = new MutationObserver(() => {
                enhanceHeader();
                enhanceContent();
            });
            accObserver.observe(acc, { childList: true, subtree: true });
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            initModal();
            initSearchObserver();
            initAccordionHeader();
            window.selectPdfResult(0);
        });
    } else {
        initModal();
        initSearchObserver();
        initAccordionHeader();
        window.selectPdfResult(0);
    }
}
