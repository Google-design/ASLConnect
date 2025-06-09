import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { NavComponent } from './components/nav/nav.component';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { ModulesComponent } from './components/modules/modules.component';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatCardModule } from '@angular/material/card';
import { MatMenuModule } from '@angular/material/menu';
import { ContactUsComponent } from './components/contact-us/contact-us.component';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatDialogModule } from '@angular/material/dialog';
import {MatBadgeModule} from '@angular/material/badge';
import {MatTabsModule} from '@angular/material/tabs';
import { MatRadioModule } from '@angular/material/radio';
import { MatExpansionModule } from '@angular/material/expansion';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatChipsModule} from '@angular/material/chips';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { NgxCaptchaModule } from 'ngx-captcha';
import { MoreResourcesComponent } from './components/more-resources/more-resources.component';
import { MatTreeModule } from '@angular/material/tree';
import { LoginComponent } from './components/login/login.component';
import { TextToAslComponent } from './components/text-to-asl/text-to-asl.component';
import { AslToTextComponent } from './components/asl-to-text/asl-to-text.component';
import { DefaultComponent } from './components/default/default.component';

// Importing the necessary Firebase modules
import { AngularFireModule } from '@angular/fire/compat';
import { AngularFireAuthModule } from '@angular/fire/compat/auth';
import { AngularFirestoreModule } from '@angular/fire/compat/firestore';
import { AngularFireDatabaseModule } from '@angular/fire/compat/database';
import { AngularFireStorageModule } from '@angular/fire/compat/storage';
import { environment } from '../environments/environment';
import { AdminComponent } from './components/admin/admin.component'; // Importing environment file
import { AngularFireAuthGuardModule } from '@angular/fire/compat/auth-guard';
import { AddModuleDialogComponent } from './components/add-module-dialog/add-module-dialog.component'; // Import AuthGuard from compat version
import { HttpClientModule } from '@angular/common/http';
import { SafeUrlPipe } from './pipes/safe-url.pipe';
import { CameraViewComponent } from './components/camera-view/camera-view.component';

@NgModule({
  declarations: [
    AppComponent,
    NavComponent,
    ModulesComponent,
    ContactUsComponent,
    MoreResourcesComponent,
    LoginComponent,
    TextToAslComponent,
    AslToTextComponent,
    DefaultComponent,
    AdminComponent,
    AddModuleDialogComponent,
    SafeUrlPipe,
    CameraViewComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    MatToolbarModule,
    MatButtonModule,
    MatSidenavModule,
    MatIconModule,
    MatListModule,
    MatGridListModule,
    HttpClientModule,
    MatCardModule,
    MatBadgeModule,
    MatMenuModule,
    MatTabsModule,
    MatDialogModule,
    MatInputModule,
    MatChipsModule,
    MatProgressSpinnerModule,
    MatSelectModule,
    MatExpansionModule,
    MatRadioModule,
    FormsModule,
    ReactiveFormsModule,
    NgxCaptchaModule,
    MatTreeModule,
    // Initialize Firebase
    AngularFireModule.initializeApp(environment.firebaseConfig),
    AngularFireAuthModule,       // Firebase Authentication
    AngularFireAuthGuardModule,
    AngularFirestoreModule,      // Firestore Database
    AngularFireDatabaseModule,   // Realtime Database
    AngularFireStorageModule,     // Firebase Storage,
  ],
  providers: [
    provideAnimationsAsync(),
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
