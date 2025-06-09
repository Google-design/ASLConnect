import { Component, OnInit, ViewChild } from '@angular/core';
import { AngularFirestore } from '@angular/fire/compat/firestore';
import { AuthService } from '../../services/auth.service';
import { Router } from '@angular/router';
import { MatTabGroup } from '@angular/material/tabs';
import { MatDialog } from '@angular/material/dialog';
import { AddModuleDialogComponent } from '../add-module-dialog/add-module-dialog.component';

@Component({
  selector: 'app-admin',
  templateUrl: './admin.component.html',
  styleUrl: './admin.component.scss',
})
export class AdminComponent implements OnInit{
  notifications: any[] = [];  //for notifications
  modules: any[] = [];    // for modules
  badgeCount: number = 0;
  showLogoutConfirm: boolean = false;
  @ViewChild('tabGroup') tabGroup: MatTabGroup; // Reference to the MatTabGroup


  constructor(
    private authService: AuthService,
    private router: Router,
    private firestore: AngularFirestore,
    private dialog: MatDialog
  ) {
    // Redirect if not logged in
    this.authService.user$.subscribe(user => {
    });
  }

  ngOnInit(): void {
    this.firestore.collection('Contact-Us').snapshotChanges().subscribe((res) => {
      this.notifications = res.map(e => {
        const data: any = e.payload.doc.data();
        return { id: e.payload.doc.id, ...data };
      });
      this.badgeCount = this.notifications.length;    //Setting the badge count based on the # of documents
    });

    this.firestore.collection('Modules', ref => ref.orderBy('id', 'asc')).snapshotChanges().subscribe((res) => {
      this.modules = res.map(e => {
        const data: any = e.payload.doc.data();
        return { id: e.payload.doc.id, ...data };
      });
    });
    
  }

  onTabChange(event: any) {
    if (event.index === 2) { // Assuming "Logout" is the third tab (index 2)
      this.showLogoutConfirm = true;
    } else {
      this.showLogoutConfirm = false;
    }
  }

  // Method to delete a notification
  deleteNotification(notificationId: string): void {
    const confirmed = window.confirm(`Are you sure you want to delete ${notificationId} message?`);

    if(confirmed){
      this.firestore.collection('Contact-Us').doc(notificationId).delete().then(() => {
        console.log('Notification successfully deleted!');
      }).catch((error) => {
        console.error('Error removing notification: ', error);
      });
    }
  }

  formatYTUrl(url?: string): string{
    if(url){
      const youtubeRegex = /^https?:\/\/(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)/;
      const match = url.match(youtubeRegex);
      if (match && match[3]) {
        console.log(match[3]);
        console.log(match);
        // If it's a YT link, transform it to the embed format
        return `https://www.youtube.com/embed/${match[3]}`;
      }
      // Return the original URL if it's not a YouTube link
      return url;
    }
    return '';   //if no link
  }

  // Method to delete a module
  deleteModule(moduleId: string): void {
    const confirmed = window.confirm(`Are you sure you want to delete ${moduleId} module?`);

    if(confirmed){
      this.firestore.collection('Modules').doc(moduleId).delete().then(() => {
        console.log('Module successfully deleted!');
      }).catch((error) => {
        console.error('Error removing module: ', error);
      });
    }
  }

  cancelLogout() {
    this.showLogoutConfirm = false;
    this.tabGroup.selectedIndex = 0;
  }

  logout() {
      this.authService.logout();
      this.showLogoutConfirm = false;
  }

  openAddModuleDialog() {
    this.dialog.open(AddModuleDialogComponent, {
    });
  }

  logout2() {
    const confirmed = window.confirm(`Are you sure you want to logout?`);
    if(confirmed)
      this.authService.logout();
  }
}
