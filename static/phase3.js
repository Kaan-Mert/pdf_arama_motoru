() => {
    window.selectPdfResult = function(idx) {
        // 1. Sol paneldeki sonuç kartlarını güncelle (yalnızca #search-results altında)
        const cards = document.querySelectorAll('#search-results .result-card');
        cards.forEach((c, i) => {
            if (i === idx) {
                c.classList.add('active-result-card');
            } else {
                c.classList.remove('active-result-card');
            }
        });

        // 2. Sağ paneldeki galeri küçük resmini (thumbnail) tıkla (yalnızca #pdf-preview-gallery altında)
        const gallery = document.querySelector('#pdf-preview-gallery');
        if (!gallery) {
            console.warn("PDF galeri kök öğesi bulunamadı (#pdf-preview-gallery)", { idx });
            return;
        }

        const thumbs = gallery.querySelectorAll('button.thumbnail-item');
        if (thumbs && thumbs[idx]) {
            thumbs[idx].click();
            thumbs[idx].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        } else {
            console.warn("PDF galeri küçük resmi bulunamadı", { idx, foundCount: thumbs ? thumbs.length : 0 });
        }
    };

    // Global tıklama dinleyicisi (Event delegation - yalnızca #search-results içindeki kartlar)
    if (!window._pdf_click_listener_bound) {
        window._pdf_click_listener_bound = true;
        document.addEventListener('click', function(e) {
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
    }

    // İlk kartı otomatik seç (yalnızca #search-results altında)
    setTimeout(() => {
        const first = document.querySelector('#search-results .result-card');
        if (first && !document.querySelector('#search-results .active-result-card')) {
            first.classList.add('active-result-card');
        }
    }, 250);
}
