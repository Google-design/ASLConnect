import { Component, inject, OnInit } from '@angular/core';

import { FormBuilder, Validators } from '@angular/forms';
import {MatSnackBar} from '@angular/material/snack-bar';
import { AngularFirestore, AngularFirestoreCollection } from '@angular/fire/compat/firestore';

@Component({
  selector: 'app-contact-us',
  templateUrl: './contact-us.component.html',
  styleUrl: './contact-us.component.scss'
})
export class ContactUsComponent implements OnInit{
  // hasUnitNumber = false;
  private _snackBar = inject(MatSnackBar);
  isSubmit: boolean = false;
  submitMessage: string = '';
  
  private fb = inject(FormBuilder);
  addressForm = this.fb.group({
    // company: null,
    firstName: [null, Validators.required],
    lastName: [null, Validators.required],
    email: [null, Validators.compose([Validators.required, Validators.email])],
    number: [null, Validators.compose([Validators.required, Validators.minLength(10)])],
    subject: [null, Validators.required],
    message: [null, Validators.required],
    // shipping: ['free', Validators.required]
  });

  private myForm: AngularFirestoreCollection<any>;

  constructor(private firestore: AngularFirestore) {

  }

  ngOnInit(): void {
    this.myForm = this.firestore.collection('Contact-Us');
  }

  onSubmit(value: any): void {
    // console.log(value);
    if (this.addressForm.valid) {
      const subject = value.subject;    //Getting the subject

      this.myForm.doc(subject).set(value).then(res => {
        this.submitMessage = 'Thank you for contacting us.';
      })
      .catch(e => {
        console.log(e);
      });

      this.isSubmit = true;
      setTimeout(() => {
        this.isSubmit = false;
      }, 5000);

      this._snackBar.open("We will reach out to you as soon as possible!","close", {duration: 4900, horizontalPosition: 'end'});
    }
    else {
      console.log("Form is invalid");
    }
  }
}
