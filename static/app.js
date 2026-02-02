/**
 * AugMentor 2.0 - Enhanced Upload Experience
 */

document.addEventListener('DOMContentLoaded', () => {
  const uploadZone = document.getElementById('uploadZone');
  const mediaInput = document.getElementById('mediaInput');
  const uploadLabel = document.getElementById('uploadLabel');
  const uploadIconEmoji = document.getElementById('uploadIconEmoji');
  const fileName = document.getElementById('fileName');
  const submitBtn = document.getElementById('submitBtn');
  const uploadForm = document.getElementById('uploadForm');

  if (!uploadZone || !mediaInput) return;

  // File type icons
  const fileIcons = {
    video: '🎬',
    image: '🖼️',
    default: '📁'
  };

  // Handle file selection
  function handleFileSelect(file) {
    if (!file) return;

    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');

    if (!isVideo && !isImage) {
      showError('Please select a valid video or image file');
      return;
    }

    // Update UI
    uploadZone.classList.add('has-file');
    uploadIconEmoji.textContent = isVideo ? fileIcons.video : fileIcons.image;
    uploadLabel.textContent = 'File selected';
    fileName.textContent = file.name;
    fileName.style.display = 'block';

    // Add subtle animation
    uploadZone.style.animation = 'none';
    uploadZone.offsetHeight; // Trigger reflow
    uploadZone.style.animation = 'pulse 0.3s ease';
  }

  // File input change
  mediaInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  });

  // Drag and drop handlers
  ['dragenter', 'dragover'].forEach(eventName => {
    uploadZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadZone.classList.remove('drag-over');
    });
  });

  uploadZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      // Create a new DataTransfer to set the file to the input
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      mediaInput.files = dataTransfer.files;
      handleFileSelect(file);
    }
  });

  // Form submission with loading state
  if (uploadForm && submitBtn) {
    uploadForm.addEventListener('submit', (e) => {
      if (!mediaInput.files || mediaInput.files.length === 0) {
        e.preventDefault();
        showError('Please select a file first');
        return;
      }

      // Show loading state
      submitBtn.disabled = true;
      submitBtn.innerHTML = `
        <span class="spinner" style="width: 20px; height: 20px; border-width: 2px;"></span>
        <span>Processing...</span>
      `;
    });
  }

  // Error display
  function showError(message) {
    // Create error toast
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      background: #ef4444;
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      font-weight: 500;
      z-index: 1000;
      animation: slideUp 0.3s ease;
    `;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = 'fadeOut 0.3s ease forwards';
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  // Add pulse animation keyframes
  const style = document.createElement('style');
  style.textContent = `
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.02); }
      100% { transform: scale(1); }
    }
    @keyframes fadeOut {
      to { opacity: 0; transform: translateX(-50%) translateY(20px); }
    }
  `;
  document.head.appendChild(style);
});

/**
 * Results page enhancements
 */
document.addEventListener('DOMContentLoaded', () => {
  // Add smooth reveal animations to result cards
  const resultCards = document.querySelectorAll('.result-card');
  
  if (resultCards.length > 0) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }
      });
    }, { threshold: 0.1 });

    resultCards.forEach(card => {
      card.style.opacity = '0';
      card.style.transform = 'translateY(20px)';
      card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      observer.observe(card);
    });

    // Trigger animation after a small delay
    setTimeout(() => {
      resultCards.forEach((card, index) => {
        setTimeout(() => {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, index * 100);
      });
    }, 100);
  }

  // Enhanced video player
  const video = document.querySelector('.media-container video');
  if (video) {
    // Add custom controls styling on play
    video.addEventListener('play', () => {
      video.parentElement.classList.add('playing');
    });
    video.addEventListener('pause', () => {
      video.parentElement.classList.remove('playing');
    });
  }
});
