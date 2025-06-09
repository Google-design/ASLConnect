import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private predictUrl = 'https://sparsh123d-myspace.hf.space/predict'; // URL to your Flask predict endpoint
  //private predictUrl = 'https://54d4-2600-6c56-4f0-890-81a6-27b1-d0e1-5aa0.ngrok-free.app/predict'; // URL to your Flask predict endpoint
  constructor(private http: HttpClient) {}

  // Updated to send an object that includes the image data
  getPrediction(payload: { image: string }): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    });
    return this.http.post<any>(this.predictUrl, payload, { headers });
  }
}
