import { Component, OnInit } from '@angular/core';
import { FormGroup, Validators, FormBuilder } from '@angular/forms';

@Component({
  selector: 'app-default',
  templateUrl: './default.component.html',
  styleUrls: ['./default.component.scss'] // Note: 'styleUrls' should be plural
})
export class DefaultComponent implements OnInit {

  constructor(private formBuilder: FormBuilder) {}

  ngOnInit() {
    
  }
}
