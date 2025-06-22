document.addEventListener('DOMContentLoaded', () => {
  // === CREATOR BLOCK ANIMATION ON SCROLL ===
  const container = document.querySelector('.creators-container');
  if (container) {
    container.classList.add('preload');

    window.addEventListener('scroll', () => {
      const triggerPoint = container.offsetTop - window.innerHeight / 1.3;
      if (window.scrollY > triggerPoint) {
        container.classList.add('active');
        container.classList.remove('preload');
      }
    });
  }

  // === FLOATING BUBBLES SCRIPT ===
  const bubbleContainer = document.querySelector(".floating-bubbles");
  if (bubbleContainer) {
    const bubbles = bubbleContainer.querySelectorAll(".bubble");
    const containerWidth = bubbleContainer.offsetWidth;
    const containerHeight = bubbleContainer.offsetHeight;

    const bubbleData = [];

    bubbles.forEach((bubble) => {
      const size = 50 + Math.random() * 20; // Limit max size to 70px
      bubble.style.width = size + "px";
      bubble.style.height = size + "px";

      const data = {
        el: bubble,
        x: Math.random() * (containerWidth - size),
        y: Math.random() * (containerHeight - size),
        dx: (Math.random() * 1.5 + 0.5) * (Math.random() < 0.5 ? 1 : -1),
        dy: (Math.random() * 1.5 + 0.5) * (Math.random() < 0.5 ? 1 : -1),
        size: size
      };

      bubble.style.left = data.x + "px";
      bubble.style.top = data.y + "px";
      bubble.style.animationDelay = `${Math.random() * 5}s`;
      bubbleData.push(data);
    });

    function animate() {
      const cw = bubbleContainer.offsetWidth;
      const ch = bubbleContainer.offsetHeight;

      bubbleData.forEach(b => {
        b.x += b.dx;
        b.y += b.dy;

        if (b.x <= 0 || b.x + b.size >= cw) b.dx *= -1;
        if (b.y <= 0 || b.y + b.size >= ch) b.dy *= -1;

        b.el.style.left = b.x + "px";
        b.el.style.top = b.y + "px";
      });

      requestAnimationFrame(animate);
    }

    animate();
  }

  // === VIDEO PLAY BUTTON FUNCTIONALITY ===
  const playButton = document.querySelector(".svg-play-button");
  const video = document.getElementById("demoVideo");

  if (playButton && video) {
    playButton.addEventListener("click", () => {
      video.play();
      video.controls = true;
      playButton.style.display = "none";
    });

    video.addEventListener("pause", () => {
      if (!video.ended) {
        playButton.style.display = "flex";
      }
    });

    video.addEventListener("play", () => {
      playButton.style.display = "none";
    });
  }

  // === SECTION SCROLL REVEAL ANIMATION ===
  const sections = document.querySelectorAll(
    "section, .prompt-section, .describe-section, .creators-section, .vibe-engine-section, .video-section, .launch-demo-section"
  );

  const revealOnScroll = () => {
    const triggerBottom = window.innerHeight * 0.85;

    sections.forEach(section => {
      const sectionTop = section.getBoundingClientRect().top;
      if (sectionTop < triggerBottom) {
        section.classList.add("in-view");
      }
    });
  };

  window.addEventListener("scroll", revealOnScroll);
  revealOnScroll(); 
});
