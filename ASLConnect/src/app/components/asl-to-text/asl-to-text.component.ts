// import { Component, OnInit } from '@angular/core';
// import { ApiService } from '../../services/api.service';

// @Component({
//   selector: 'app-asl-to-text',
//   templateUrl: './asl-to-text.component.html',
//   styleUrls: ['./asl-to-text.component.scss']
// })
// export class AslToTextComponent implements OnInit {
//   video: any;  // To reference video element
//   canvas: HTMLCanvasElement;
//   ctx: CanvasRenderingContext2D;
//   prediction: string = ''; // To store the predicted ASL letter

//   constructor(private apiService: ApiService) {}

//   ngOnInit(): void {
//     this.setupCamera();
//   }

//   // Initialize the camera and start capturing video
//   setupCamera() {
//     this.video = document.getElementById('videoElement') as HTMLVideoElement;
//     this.canvas = document.createElement('canvas');
//     this.ctx = this.canvas.getContext('2d')!;
    
//     // Access the camera and stream video to the <video> element
//     navigator.mediaDevices.getUserMedia({ video: true })
//       .then((stream) => {
//         this.video.srcObject = stream;
//       })
//       .catch((err) => {
//         console.error("Error accessing camera: ", err);
//       });

//     // Start capturing frames
//     setInterval(() => this.captureFrame(), 100);  // Capture frame every 100ms
//   }

//   // Capture a frame from the video stream and include a timestamp
//   captureFrame() {
//     // Draw the video frame to canvas
//     this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

//     // Convert canvas content to a base64 JPEG image
//     const frameData = this.canvas.toDataURL('image/jpeg');
//     const timestamp = Date.now(); // Get the current timestamp in milliseconds

//     // Construct the payload with the image data and timestamp
//     const payload = {
//       image: frameData,
//       timestamp: timestamp
//     };

//     // Send captured frame and timestamp to Flask backend for prediction
//     this.apiService.getPrediction(payload).subscribe((response: any) => {
//       this.prediction = response.prediction;  // Display the prediction
//       console.log("Predicted ASL character: ", this.prediction);
//     });
//   }
// }

import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-asl-to-text',
  templateUrl: './asl-to-text.component.html',
  styleUrls: ['./asl-to-text.component.scss']
})
export class AslToTextComponent implements OnInit {
  constructor() {}

  ngOnInit(): void {
    // Initialization logic can go here if needed
  }

  // Any additional methods specific to the text conversion info or navigation can go here
}