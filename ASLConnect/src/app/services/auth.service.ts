import { Injectable } from '@angular/core';
import { Auth, signInWithEmailAndPassword, signOut, sendPasswordResetEmail } from '@angular/fire/auth';
import { AngularFireAuth } from '@angular/fire/compat/auth';
import { Router } from '@angular/router';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  user$: Observable<any>; // Observable for auth state

  constructor(private auth: AngularFireAuth, private router: Router) {
    this.user$ = this.auth.authState;  // Track the authentication state
  }

  async login( {email, password}: {email: string, password: string} ) {    //changed the parameters type
    try {
      const user = await this.auth.signInWithEmailAndPassword(
        email,
        password
      );
      return user;
    } catch (e) {
      return null;
    }
  }

  logout() {
    return this.auth.signOut().then(() => {
      this.router.navigateByUrl('/login');
    });  }

  async sendPasswordResetEmail(email: string): Promise<void> {
    try {
      await this.auth.sendPasswordResetEmail(email);
    } catch (error: any) {
      console.error("Error sending password reset email", error);
      throw error;
    }
  }
}
