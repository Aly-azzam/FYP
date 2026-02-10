/**
 * AugMentor 2.0 — App Logic
 */

// ===== Dark Mode Persistence (all pages) =====
document.addEventListener('DOMContentLoaded', () => {
  if (localStorage.getItem('augmentor_dark_mode') === 'true') {
    document.body.classList.add('dark-mode');
  }
});


// ===== Upload Page =====
document.addEventListener('DOMContentLoaded', () => {
  const uploadZone = document.getElementById('uploadZone');
  const mediaInput = document.getElementById('mediaInput');
  const uploadLabel = document.getElementById('uploadLabel');
  const uploadIconEmoji = document.getElementById('uploadIconEmoji');
  const fileName = document.getElementById('fileName');
  const submitBtn = document.getElementById('submitBtn');
  const uploadForm = document.getElementById('uploadForm');

  if (!uploadZone || !mediaInput) return;

  const fileIcons = { video: '🎬', image: '🖼️', default: '📁' };

  // Preview elements
  const filePreview = document.getElementById('filePreview');
  const filePreviewThumb = document.getElementById('filePreviewThumb');
  const filePreviewName = document.getElementById('filePreviewName');
  const fileTypeBadge = document.getElementById('fileTypeBadge');
  const fileSizeEl = document.getElementById('fileSize');
  const fileDurationEl = document.getElementById('fileDuration');
  const clearFileBtn = document.getElementById('clearFileBtn');
  const clearAllBtn = document.getElementById('clearAllBtn');

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function handleFileSelect(file) {
    if (!file) return;
    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');

    if (!isVideo && !isImage) {
      showToast('Please select a valid video or image file', 'error');
      return;
    }

    // Upload zone state
    uploadZone.classList.add('has-file');
    uploadIconEmoji.textContent = isVideo ? fileIcons.video : fileIcons.image;
    uploadLabel.textContent = 'File selected';
    fileName.textContent = file.name;
    fileName.style.display = 'block';

    // Show file preview card
    if (filePreview) {
      filePreview.classList.add('visible');
      filePreviewName.textContent = file.name;
      fileSizeEl.textContent = formatFileSize(file.size);

      // Type badge
      fileTypeBadge.textContent = isVideo ? 'VIDEO' : 'IMAGE';
      fileTypeBadge.className = 'file-type-badge ' + (isVideo ? 'video' : 'image');

      // Thumbnail
      filePreviewThumb.innerHTML = '';
      const url = URL.createObjectURL(file);

      if (isImage) {
        const img = document.createElement('img');
        img.src = url;
        filePreviewThumb.appendChild(img);
        fileDurationEl.textContent = '';
      } else {
        const vid = document.createElement('video');
        vid.src = url;
        vid.muted = true;
        vid.preload = 'metadata';
        vid.addEventListener('loadedmetadata', () => {
          const dur = vid.duration;
          if (dur && isFinite(dur)) {
            const mins = Math.floor(dur / 60);
            const secs = Math.floor(dur % 60);
            fileDurationEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
          }
          // Seek a bit for thumbnail
          vid.currentTime = Math.min(1, dur * 0.1);
        });
        vid.addEventListener('seeked', () => {
          const canvas = document.createElement('canvas');
          canvas.width = 120;
          canvas.height = 120;
          canvas.getContext('2d').drawImage(vid, 0, 0, canvas.width, canvas.height);
          const img = document.createElement('img');
          img.src = canvas.toDataURL();
          filePreviewThumb.innerHTML = '';
          filePreviewThumb.appendChild(img);
        });
        filePreviewThumb.appendChild(vid);
      }
    }
  }

  // Clear file
  function clearFile() {
    mediaInput.value = '';
    uploadZone.classList.remove('has-file');
    uploadIconEmoji.textContent = '⬆️';
    uploadLabel.textContent = 'Select Video to Upload';
    fileName.style.display = 'none';
    fileName.textContent = '';
    if (filePreview) filePreview.classList.remove('visible');
    if (fileDurationEl) fileDurationEl.textContent = '';
  }

  if (clearFileBtn) clearFileBtn.addEventListener('click', clearFile);
  if (clearAllBtn) clearAllBtn.addEventListener('click', () => {
    clearFile();
    // Reset toggles to checked
    document.querySelectorAll('.toggle-card input[type="checkbox"]').forEach(cb => cb.checked = true);
  });

  mediaInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
  });

  // Drag and drop
  ['dragenter', 'dragover'].forEach(ev => {
    uploadZone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      uploadZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(ev => {
    uploadZone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      uploadZone.classList.remove('drag-over');
    });
  });

  uploadZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      mediaInput.files = dt.files;
      handleFileSelect(file);
    }
  });

  // Form submit → processing overlay
  if (uploadForm && submitBtn) {
    uploadForm.addEventListener('submit', (e) => {
      if (!mediaInput.files || mediaInput.files.length === 0) {
        e.preventDefault();
        showToast('Please select a file first', 'error');
        return;
      }

      // Show processing overlay
      const overlay = document.getElementById('processingOverlay');
      if (overlay) overlay.classList.add('active');

      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner"></span> <span>Processing...</span>';
    });
  }
});


// ===== Results Page Enhancements =====
document.addEventListener('DOMContentLoaded', () => {
  const resultCards = document.querySelectorAll('.result-card');

  if (resultCards.length > 0) {
    setTimeout(() => {
      resultCards.forEach((card, index) => {
        setTimeout(() => {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, index * 80);
      });
    }, 100);
  }
});


// ===== Toast Notification =====
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.textContent = message;

  const colors = {
    error: '#ef4444',
    success: '#10b981',
    info: '#6366f1',
  };

  toast.style.cssText = `
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    background: ${colors[type] || colors.info};
    color: white;
    padding: 12px 24px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.9rem;
    z-index: 10000;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    animation: toastIn 0.3s ease;
  `;

  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// Inject toast animation
const toastStyle = document.createElement('style');
toastStyle.textContent = `
  @keyframes toastIn {
    from { opacity: 0; transform: translateX(-50%) translateY(20px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
  }
`;
document.head.appendChild(toastStyle);
