document.addEventListener("DOMContentLoaded", () => {
  new App();
});

class App {
  constructor() {
    const video = document.querySelector("#video");

    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          // function 의 this와 화살표 함수의 this 가 다름
          video.srcObject = stream;
        })
        .catch(function (error) {
          console.log("Something went wrong!");
          console.log(error);
          return;
        });
    }

    video.addEventListener("loadedmetadata", () => {
      window.requestAnimationFrame(this.draw.bind(this));
    });
  }
}
