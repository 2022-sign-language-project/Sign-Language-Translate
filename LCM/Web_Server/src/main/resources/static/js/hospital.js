const video = document.querySelector(".test_video");
let videoMediaStream = null;

navigator.mediaDevices
.getUserMedia({
    audio: false,
    video: {
        width:360,
        height:240,
    }
})
.then((stream) => {

  video.srcObject = stream;
  video.onloadedmetadata = function(){
    video.play();
  };
  videoMediaStream = stream;
  console.log(stream);
})
.catch(function (error) {
  console.log("Something went wrong!");
  console.log(error);
  return;
});



