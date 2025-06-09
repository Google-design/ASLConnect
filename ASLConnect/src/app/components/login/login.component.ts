import { Component, inject, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AuthService } from '../../services/auth.service';
import { Router } from '@angular/router';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent implements OnInit{
  private _snackBar = inject(MatSnackBar);
  public aFormGroup: FormGroup;
  siteKey: string = '6LebPWYqAAAAAGf7X3SD7lZ0lDE9_B4toDkpM2Do'; // Declare siteKey correctly

  credentials: FormGroup;
  isLoading = false;
  loginFailed: boolean = false;

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router,
  ) { }

  get email() {
    return this.credentials.get('email')!;
  }

  get password() {
    return this.credentials.get('password')!;
  }

  ngOnInit(): void {
    this.credentials = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]]
    });

    this.aFormGroup = this.fb.group({
      recaptcha: ['', Validators.required]
    });
  }

  async login() {
    const user = await this.authService.login(this.credentials.value);
    if(user){
      this.router.navigateByUrl('/admin');
      console.log("LOGIN SUCCESSFULL!");

    } else {
      // this._snackBar.open("Invalid email or password","close", {duration: 4900, horizontalPosition: 'end'});
      this.loginFailed = true;
      console.log("LOGIN FAILED!");
    }
  }

  async forgotPassword() {
    const email = this.credentials.value.email;
    if(email){
      this.authService.sendPasswordResetEmail(email)
        .then(() => {
          this._snackBar.open("Password Reset email has been sent!","close", {duration: 4900, horizontalPosition: 'end'});
          console.log("Password Reset email has been sent!");
        })
        .catch((error: any) => {
          // this.showAlert('Error', 'Failed to send password reset email. Please try again later.');
          this._snackBar.open("Failed to send password reset email. Please try again with valid email address.","close", {duration: 4900, horizontalPosition: 'end'});
          console.error('Error sending password reset email:', error);
        });
    } else {
      this._snackBar.open("Email has to be provided!","close", {duration: 4900, horizontalPosition: 'end'});
      console.log("ERROR: Email has to be provided!");
      // this.showAlert("Error", "Please provide the email");
    }
  }
}
