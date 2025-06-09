import { Component } from '@angular/core';
import { ApiService } from './services/api.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'ASLConnect';
  // data: any;

  // constructor(private apiService: ApiService) {}

  // ngOnInit(): void {
  //   this.apiService.getData().subscribe(response => {
  //     this.data = response;
  //     console.log(this.data);
  //   });
  // }
}
