/**
 * Canopy.AI — Bengaluru Urban Green Space Landing Page Logic
 * Interactive GIS grid, NDVI sensitivity simulation, tile explorer & modal
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Navbar scroll effect
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });

  // 2. Fetch or load tile data
  let allTiles = [];

  fetch('tile_data.json')
    .then(res => res.json())
    .then(data => {
      allTiles = data;
      initGISGrid(allTiles);
      initTileGallery(allTiles);
      initNdviSimulator(allTiles);
    })
    .catch(err => {
      console.warn('Could not load tile_data.json via fetch, using default sample set:', err);
      // Fallback sample set in case opened directly via file:// protocol
      allTiles = generateSampleTiles();
      initGISGrid(allTiles);
      initTileGallery(allTiles);
      initNdviSimulator(allTiles);
    });

  // 3. Simulated GIS Grid
  const gisGrid = document.getElementById('gisGrid');
  const inspCard = document.getElementById('inspectorCard');
  const inspTileId = document.getElementById('inspTileId');
  const inspClass = document.getElementById('inspClass');
  const inspConf = document.getElementById('inspConf');
  const inspNdvi = document.getElementById('inspNdvi');

  let currentLayer = 'pred'; // 'pred', 'ndvi', 'sat'

  function initGISGrid(tiles) {
    if (!gisGrid) return;
    gisGrid.innerHTML = '';

    tiles.forEach(tile => {
      const cell = document.createElement('div');
      cell.className = 'gis-tile-cell';
      cell.dataset.id = tile.id;
      cell.dataset.label = tile.label;
      cell.dataset.ndvi = tile.ndvi;
      cell.dataset.conf = tile.conf;

      applyLayerColor(cell, tile, currentLayer);

      cell.addEventListener('mouseenter', () => updateInspector(tile));
      cell.addEventListener('click', () => openTileModal(tile));

      gisGrid.appendChild(cell);
    });
  }

  function applyLayerColor(element, tile, layer) {
    if (layer === 'pred') {
      if (tile.label === 'green') {
        element.style.background = '#22c55e';
        element.style.border = '1px solid #16a34a';
        element.style.boxShadow = '0 0 8px rgba(34, 197, 94, 0.4)';
      } else {
        element.style.background = '#475569';
        element.style.border = '1px solid #334155';
        element.style.boxShadow = 'none';
      }
    } else if (layer === 'ndvi') {
      const v = Math.max(0, Math.min(1, (tile.ndvi + 0.1) / 0.7)); // scale ~ -0.1 to 0.6
      const r = Math.round(255 * (1 - v));
      const g = Math.round(255 * v);
      element.style.background = `rgb(${r}, ${g}, 40)`;
      element.style.border = '1px solid rgba(255,255,255,0.15)';
      element.style.boxShadow = v > 0.5 ? '0 0 6px rgba(34, 197, 94, 0.3)' : 'none';
    } else if (layer === 'sat') {
      // Natural satellite RGB appearance
      if (tile.label === 'green') {
        element.style.background = '#1a472a';
        element.style.border = '1px solid #2d6a4f';
      } else {
        element.style.background = '#6c757d';
        element.style.border = '1px solid #495057';
      }
      element.style.boxShadow = 'none';
    }
  }

  function updateInspector(tile) {
    if (!inspCard) return;
    inspTileId.textContent = tile.id;
    inspClass.textContent = tile.label === 'green' ? 'Green Canopy' : 'Urban / Impervious';
    inspClass.className = tile.label === 'green' ? 'text-green' : '';
    inspConf.textContent = `${tile.conf}%`;
    inspNdvi.textContent = tile.ndvi.toFixed(4);

    const badge = inspCard.querySelector('.inspector-badge');
    if (badge) {
      if (tile.label === 'green') {
        badge.textContent = 'Lush Canopy Detected';
        badge.className = 'inspector-badge green-badge';
      } else {
        badge.textContent = 'Urban Surface';
        badge.className = 'inspector-badge slate-badge';
      }
    }
  }

  // Layer buttons
  const btnPred = document.getElementById('btnPredLayer');
  const btnNdvi = document.getElementById('btnNdviLayer');
  const btnSat = document.getElementById('btnSatLayer');
  const layerBtns = [btnPred, btnNdvi, btnSat];

  function setLayer(layer, activeBtn) {
    currentLayer = layer;
    layerBtns.forEach(b => b && b.classList.remove('active'));
    if (activeBtn) activeBtn.classList.add('active');

    const cells = document.querySelectorAll('.gis-tile-cell');
    cells.forEach(cell => {
      const tile = allTiles.find(t => t.id === cell.dataset.id);
      if (tile) applyLayerColor(cell, tile, currentLayer);
    });
  }

  if (btnPred) btnPred.addEventListener('click', () => setLayer('pred', btnPred));
  if (btnNdvi) btnNdvi.addEventListener('click', () => setLayer('ndvi', btnNdvi));
  if (btnSat) btnSat.addEventListener('click', () => setLayer('sat', btnSat));

  // 4. Tile Gallery Section
  const gallery = document.getElementById('tileGallery');
  const filterBtns = document.querySelectorAll('.filter-btn');

  function initTileGallery(tiles) {
    if (!gallery) return;
    renderGallery(tiles.slice(0, 16)); // show top 16 curated tiles
  }

  function renderGallery(tiles) {
    gallery.innerHTML = '';
    tiles.forEach(tile => {
      const isGreen = tile.label === 'green';
      const card = document.createElement('div');
      card.className = 'tile-gallery-card';

      // Use copied previews or fallback
      const imgSrc = `assets/tiles/${tile.id}.png`;

      card.innerHTML = `
        <div class="tile-img-container">
          <img src="${imgSrc}" alt="${tile.id}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'200\\' height=\\'200\\' viewBox=\\'0 0 200 200\\'><rect width=\\'200\\' height=\\'200\\' fill=\\'${isGreen ? '%231a472a' : '%23475569'}\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'white\\' font-size=\\'12\\' text-anchor=\\'middle\\' dominant-baseline=\\'middle\\'>${tile.id}</text></svg>'">
          <span class="tile-overlay-chip ${isGreen ? 'chip-green' : 'chip-slate'}">
            ${isGreen ? 'Green Space' : 'Urban Surface'}
          </span>
        </div>
        <div class="tile-card-info">
          <div class="tile-card-name">${tile.id}</div>
          <div class="tile-card-stats">
            <span>NDVI: <strong>${tile.ndvi.toFixed(3)}</strong></span>
            <span>Conf: <strong>${tile.conf}%</strong></span>
          </div>
        </div>
      `;

      card.addEventListener('click', () => openTileModal(tile));
      gallery.appendChild(card);
    });
  }

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.dataset.filter;
      let filtered = allTiles;
      if (filter === 'green') {
        filtered = allTiles.filter(t => t.label === 'green');
      } else if (filter === 'non_green') {
        filtered = allTiles.filter(t => t.label === 'non_green');
      }
      renderGallery(filtered.slice(0, 16));
    });
  });

  // 5. NDVI Sensitivity Simulator
  const ndviSlider = document.getElementById('ndviSlider');
  const threshValDisplay = document.getElementById('threshValDisplay');
  const simGreenCount = document.getElementById('simGreenCount');
  const simAreaKm2 = document.getElementById('simAreaKm2');
  const simStatus = document.getElementById('simStatus');

  function initNdviSimulator(tiles) {
    if (!ndviSlider) return;

    ndviSlider.addEventListener('input', (e) => {
      const thresh = parseFloat(e.target.value);
      threshValDisplay.textContent = thresh.toFixed(2);

      const greenTiles = tiles.filter(t => t.ndvi >= thresh);
      const greenCount = greenTiles.length;
      const pct = ((greenCount / tiles.length) * 100).toFixed(1);

      // 256px * 10m = 2560m = 2.56 km. Area of tile = 2.56 * 2.56 = 6.5536 km2
      const area = (greenCount * 6.5536).toFixed(1);

      simGreenCount.textContent = `${greenCount} / ${tiles.length} (${pct}%)`;
      simAreaKm2.innerHTML = `${area} km&sup2;`;

      if (thresh < 0.25) {
        simStatus.textContent = 'Permissive (Includes scrub & sparse lawns)';
        simStatus.className = 'badge-pill';
        simStatus.style.background = 'rgba(234, 179, 8, 0.15)';
        simStatus.style.color = '#facc15';
      } else if (thresh > 0.38) {
        simStatus.textContent = 'Strict (Dense evergreen forest only)';
        simStatus.className = 'badge-pill';
        simStatus.style.background = 'rgba(56, 189, 248, 0.15)';
        simStatus.style.color = '#38bdf8';
      } else {
        simStatus.textContent = 'Optimal (Standard Sentinel-2 SR)';
        simStatus.className = 'badge-pill optimal';
        simStatus.style.background = 'rgba(52, 211, 153, 0.15)';
        simStatus.style.color = '#34d399';
      }
    });
  }

  // 6. Evaluation Plots Tab Switcher
  const plotTabs = document.querySelectorAll('.plot-tab');
  const paneEvalReport = document.getElementById('paneEvalReport');
  const paneTrainingCurves = document.getElementById('paneTrainingCurves');

  plotTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      plotTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const target = tab.dataset.tab;
      if (target === 'evalReport') {
        paneEvalReport.classList.add('active');
        paneTrainingCurves.classList.remove('active');
      } else {
        paneEvalReport.classList.remove('active');
        paneTrainingCurves.classList.add('active');
      }
    });
  });

  // 7. Tile Modal
  const modal = document.getElementById('tileModal');
  const modalClose = document.getElementById('modalClose');
  const modalImg = document.getElementById('modalImg');
  const modalBadge = document.getElementById('modalBadge');
  const modalTileId = document.getElementById('modalTileId');
  const modalCaption = document.getElementById('modalCaption');
  const modalPred = document.getElementById('modalPred');
  const modalTrue = document.getElementById('modalTrue');
  const modalConf = document.getElementById('modalConf');
  const modalNdvi = document.getElementById('modalNdvi');
  const modalGeo = document.getElementById('modalGeo');

  function openTileModal(tile) {
    if (!modal) return;
    const isGreen = tile.label === 'green';
    modalTileId.textContent = tile.id;
    modalImg.src = `assets/tiles/${tile.id}.png`;
    modalBadge.textContent = isGreen ? 'Green Canopy' : 'Urban Impervious';
    modalBadge.style.background = isGreen ? 'rgba(34, 197, 94, 0.9)' : 'rgba(71, 85, 105, 0.9)';
    modalBadge.style.color = isGreen ? '#022c22' : '#ffffff';

    modalCaption.textContent = isGreen
      ? 'High photosynthetic activity and dense near-infrared reflectance signature.'
      : 'Asphalt, concrete, rooftop structures, or cleared urban land.';

    modalPred.textContent = isGreen ? 'Green Space' : 'Urban / Non-Green';
    modalPred.className = isGreen ? 'text-green' : '';
    modalTrue.textContent = (tile.true_label || tile.label) === 'green' ? 'Green Space' : 'Urban / Non-Green';
    modalConf.textContent = `${tile.conf}%`;
    modalNdvi.textContent = tile.ndvi.toFixed(4);
    modalGeo.textContent = `${tile.left || '77.52'}\u00B0E, ${tile.top || '13.04'}\u00B0N`;

    modal.classList.add('open');
  }

  if (modalClose) {
    modalClose.addEventListener('click', () => modal.classList.remove('open'));
  }
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('open');
    });
  }

  // 8. Fallback generator in case of direct local file usage
  function generateSampleTiles() {
    const samples = [];
    const greenIds = [
      'tile_00000_00000', 'tile_00000_00256', 'tile_00000_00512',
      'tile_00256_00000', 'tile_00256_00256'
    ];
    for (let i = 0; i < 169; i++) {
      const r = Math.floor(i / 13) * 256;
      const c = (i % 13) * 256;
      const id = `tile_${String(r).padStart(5, '0')}_${String(c).padStart(5, '0')}`;
      const isGreen = greenIds.includes(id) || (i % 7 === 0);
      samples.push({
        id,
        label: isGreen ? 'green' : 'non_green',
        true_label: isGreen ? 'green' : 'non_green',
        conf: isGreen ? 99.8 : 98.4,
        ndvi: isGreen ? 0.342 : 0.245,
        correct: true,
        left: 77.45 + (c / 10000),
        top: 13.14 - (r / 10000)
      });
    }
    return samples;
  }
});
