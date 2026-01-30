const input = document.getElementById("mediaInput");
const label = document.getElementById("fileLabel");

if (input && label) {
  input.addEventListener("change", () => {
    if (input.files && input.files.length > 0) {
      label.textContent = "📄 " + input.files[0].name;
    } else {
      label.textContent = "📂 Choose a video or image";
    }
  });
}
