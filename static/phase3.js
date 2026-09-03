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

    // --- 7. UNIFIED ACCORDION HEADER ENHANCEMENT ---
    function initAccordionHeader() {
        function enhance() {
            const acc = document.getElementById('library-accordion');
            if (!acc) return false;
            const labelWrap = acc.querySelector('.label-wrap');
            if (!labelWrap) return false;

            // Check if brandContainer already exists
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

            // Hide only the default title text span, preserve .icon / svg chevron
            const spans = labelWrap.querySelectorAll('span');
            spans.forEach(s => {
                if (
                    !s.closest('.accordion-brand-container') &&
                    !s.classList.contains('icon') &&
                    !s.querySelector('svg')
                ) {
                    s.style.display = 'none';
                }
            });

            // Ensure the chevron .icon is visible and positioned at the far right
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

            return true;
        }

        if (!enhance()) {
            let attempts = 0;
            const timer = setInterval(() => {
                attempts++;
                if (enhance() || attempts > 20) {
                    clearInterval(timer);
                }
            }, 100);
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
