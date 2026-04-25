document.addEventListener("DOMContentLoaded", function() {
    // Sidebar Toggle for Mobile
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('toggle-btn');
    
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', function() {
            sidebar.classList.toggle('show');
        });
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        if (window.innerWidth <= 991 && sidebar && sidebar.classList.contains('show')) {
            if (!sidebar.contains(event.target) && !toggleBtn.contains(event.target)) {
                sidebar.classList.remove('show');
            }
        }
    });

    // File Upload Drag & Drop functionality
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(area => {
        const fileInput = area.querySelector('input[type="file"]');
        
        area.addEventListener('click', (e) => {
            // Only trigger click if the user didn't click on a button or the input itself
            if (e.target !== fileInput && !e.target.closest('button')) {
                fileInput.click();
            }
        });

        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });

        area.addEventListener('dragleave', () => {
            area.classList.remove('dragover');
        });

        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                // Important: dispatch change event so inline handlers like previewMedia(event) trigger
                fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                updateFileName(area, e.dataTransfer.files[0].name);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                updateFileName(area, fileInput.files[0].name);
            }
        });
    });

    function updateFileName(area, name) {
        let p = area.querySelector('p');
        if(!p) {
            p = document.createElement('p');
            p.className = 'mt-3 text-muted';
            area.appendChild(p);
        }
        p.innerHTML = `<strong>Selected File:</strong> ${name}`;
    }

    // Loading Overlay functionality for forms
    const forms = document.querySelectorAll('form.needs-loading');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            if(loadingOverlay) {
                loadingOverlay.classList.add('active');
            }
        });
    });
});