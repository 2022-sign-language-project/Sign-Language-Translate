document.addEventListener("DOMContentLoaded", () => {
  new App();
});

class App {
  constructor() {
    const video = document.querySelector("#videoElement");
    let videoStream = null;

    const startBtn = document.querySelector("#start_btn");
    const endBtn = document.querySelector("#end_btn");
    const downloadBtn = document.querySelector("#download-btn");

    let mediaRecorder = null;
    let recordedMediaUrl = null;

    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ audio: false, video: { width: 700, height: 600 } })
        .then((stream) => {
          // function 의 this와 화살표 함수의 this 가 다름
          video.srcObject = stream;
          videoStream = stream;
          console.log(videoStream);
        })
        .catch(function (error) {
          console.log("Something went wrong!");
          console.log(error);
          return;
        });

      // 녹화 시작 버튼 클릭 시 빌생하는 이벤트 핸들러 등록
      startBtn.addEventListener("click", function () {
        let mediaData = [];

        // 1.MediaStream을 매개변수로 MediaRecorder 생성자를 호출
        mediaRecorder = new MediaRecorder(videoStream, {
          mimeType: "video/webm; codecs=vp8",
        });

        // 2. 전달받는 데이터를 처리하는 이벤트 핸들러 등록
        mediaRecorder.ondataavailable = function (event) {
          if (event.data && event.data.size > 0) {
            mediaData.push(event.data);
          }
        };

        // 3. 녹화 중지 이벤트 핸들러 등록
        mediaRecorder.onstop = function () {
          const blob = new Blob(mediaData, { type: "video/webm" });
          recordedMediaUrl = URL.createObjectURL(blob);
          console.log(recordedMediaUrl);

          //ajax를 이용한 server로 POST
          sendVideo(blob);
        };

        // 4. 녹화 시작
        mediaRecorder.start();
      });

      // 녹화 종료 버튼 클릭 시 빌생하는 이벤트 핸들러 등록
      endBtn.addEventListener("click", function () {
        if (mediaRecorder) {
          // 5. 녹화 중지
          mediaRecorder.stop();
          mediaRecorder = null;
        }
      });

      downloadBtn.addEventListener("click", function () {
        if (recordedMediaUrl) {
          const link = document.createElement("a");
          document.body.appendChild(link);
          // 녹화된 영상의 URL을 href 속성으로 설정
          link.href = recordedMediaUrl;
          // 저장할 파일명 설정
          link.download = "video.webm";
          link.click();
          document.body.removeChild(link);
        }
      });
    }

    const sendVideo = (blob) => {
      if (blob == null) return;

      let filename = new Date().toString() + ".avi";
      // let filename = "file" + ".avi";
      const file = new File([blob], filename);

      let fd = new FormData();
      fd.append("fname", filename);
      fd.append("file", file);

      $.ajax({
        // url: "http://58.233.13.150:5500/",
        url: "http://127.0.0.1:5500/",
        type: "POST",
        contentType: false,
        processData: false,
        data: fd,
        success: function (data, textStatus) {
          if (data != null) {
            // setUserResponse(data);
            // send(data);
            console.log("POST성공");
          }
        },
        error: function (errorMessage) {
          // setUserResponse("");
          console.log("Error" + errorMessage);
        },
      }).done((data) => {
        console.log(data);
      });
    };
  }
}
