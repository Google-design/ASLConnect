import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { MatSnackBar } from '@angular/material/snack-bar';
import { AuthService } from '../../services/auth.service';
import { Router } from '@angular/router';
import { LoginComponent } from './login.component';
import { of, throwError } from 'rxjs';

describe('LoginComponent', () => {
  let component: LoginComponent;
  let fixture: ComponentFixture<LoginComponent>;
  let mockAuthService: jasmine.SpyObj<AuthService>;
  let mockRouter: jasmine.SpyObj<Router>;
  let mockSnackBar: jasmine.SpyObj<MatSnackBar>;

  beforeEach(async () => {
    mockAuthService = jasmine.createSpyObj('AuthService', ['login', 'sendPasswordResetEmail']);
    mockRouter = jasmine.createSpyObj('Router', ['navigateByUrl']);
    mockSnackBar = jasmine.createSpyObj('MatSnackBar', ['open']);

    await TestBed.configureTestingModule({
      declarations: [LoginComponent],
      imports: [ReactiveFormsModule],
      providers: [
        FormBuilder,
        { provide: AuthService, useValue: mockAuthService },
        { provide: Router, useValue: mockRouter },
        { provide: MatSnackBar, useValue: mockSnackBar },
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(LoginComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create the login component', () => {
    expect(component).toBeTruthy();
  });

  it('should initialize the login form with empty fields', () => {
    const credentials = component.credentials.value;
    expect(credentials.email).toBe('');
    expect(credentials.password).toBe('');
  });

  it('should mark the form as invalid if fields are empty', () => {
    component.credentials.controls['email'].setValue('');
    component.credentials.controls['password'].setValue('');
    expect(component.credentials.valid).toBeFalse();
  });

  it('should mark the form as valid if valid email and password are provided', () => {
    component.credentials.controls['email'].setValue('test@example.com');
    component.credentials.controls['password'].setValue('123456');
    expect(component.credentials.valid).toBeTrue();
  });

  it('should mark recaptcha as required and invalid when empty', () => {
    component.aFormGroup.controls['recaptcha'].setValue('');
    expect(component.aFormGroup.valid).toBeFalse();
  });
  
  it('should mark recaptcha as required and invalid when empty', () => {
    component.aFormGroup.controls['recaptcha'].setValue('');
    expect(component.aFormGroup.valid).toBeFalse();
  });

  it('should send password reset email on valid email', async () => {
    mockAuthService.sendPasswordResetEmail.and.returnValue(Promise.resolve());

    component.credentials.controls['email'].setValue('test@example.com');
    await component.forgotPassword();

    expect(mockSnackBar.open).toHaveBeenCalledWith(
      'Password Reset email has been sent!',
      'close',
      { duration: 4900, horizontalPosition: 'end' }
    );
  });

  it('should display error on failed password reset email', async () => {
    mockAuthService.sendPasswordResetEmail.and.returnValue(Promise.reject('Error'));

    component.credentials.controls['email'].setValue('test@example.com');
    await component.forgotPassword();

    expect(mockSnackBar.open).toHaveBeenCalledWith(
      'Failed to send password reset email. Please try again with valid email address.',
      'close',
      { duration: 4900, horizontalPosition: 'end' }
    );
  });

  it('should display error if email is not provided for password reset', async () => {
    component.credentials.controls['email'].setValue('');
    await component.forgotPassword();

    expect(mockSnackBar.open).toHaveBeenCalledWith(
      'Email has to be provided!',
      'close',
      { duration: 4900, horizontalPosition: 'end' }
    );
  });

  it('should display a snackbar with appropriate message', () => {
    component['_snackBar'].open('Test message', 'close', { duration: 4900 });

    expect(mockSnackBar.open).toHaveBeenCalledWith('Test message', 'close', { duration: 4900 });
  });
});
