import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-asl-to-text',
  templateUrl: './camera-view.component.html',
  styleUrls: ['./camera-view.component.scss']
})
export class CameraViewComponent implements OnInit, OnDestroy {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  prediction: string = '';
  private stream: MediaStream;
  private capturing: boolean = true; // Flag to control frame capturing
  private captureInterval: any;
  constructor(private apiService: ApiService, private router: Router) {}

  ngOnInit(): void {
    this.setupCamera();
  }

  ngOnDestroy(): void {
    this.stopCamera();
  }

  setupCamera() {
    this.video = document.getElementById('videoElement') as HTMLVideoElement;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;

    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        this.stream = stream;
        this.video.srcObject = stream;
        this.startCapturingFrames(); // Start capturing frames when the stream is ready
      })
      .catch(err => {
        console.error("Error accessing camera: ", err);
      });
  }

  startCapturingFrames() {
    this.captureInterval = setInterval(() => {
      if (this.capturing) {
        this.captureFrame(); // Directly call captureFrame without timestamp
      }
    }, 100); // Consistent interval of 100ms
  }

  stopCamera() {
    if (this.video.srcObject) {
      (this.video.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      this.video.srcObject = null;
      this.capturing = false; // Stop capturing frames
      clearInterval(this.captureInterval); // Clear the interval
    }
    this.router.navigate(['../']); // Navigate back
  }

  captureFrame() {
    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    const frameData = this.canvas.toDataURL('image/jpeg');
  
    const payload = { image: frameData }; // Removed timestamp
    this.apiService.getPrediction(payload).subscribe({
      next: (response: any) => {
        this.prediction = response.prediction;
        console.log("Predicted ASL character: ", this.prediction);
      },
      error: (error: any) => {
        console.error("Error sending frame for prediction: ", error);
      }
    });
  }
}
