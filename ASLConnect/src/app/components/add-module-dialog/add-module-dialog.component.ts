import { Component } from '@angular/core';
import { AngularFirestore } from '@angular/fire/compat/firestore';
import { MatDialogRef } from '@angular/material/dialog';

@Component({
  selector: 'app-add-module-dialog',
  templateUrl: './add-module-dialog.component.html',
  styleUrl: './add-module-dialog.component.scss'
})
export class AddModuleDialogComponent {
  moduleName: string = '';
  moduleDescription: string = '';
  id: number;
  sections = [{ heading: '', content: '' }];    //Sections for subheadings
  resourceLink: string = '';
  resources = ['']; // Initialize with an empty field
  videoUrl: string = '';
  imageUrl: string = '';

  constructor(
    private dialogRef: MatDialogRef<AddModuleDialogComponent>,
    private firestore: AngularFirestore
  ) {}

  addSection() {
    this.sections.push({ heading: '', content: ''});
  }

  removeSection(index: number) {
    this.sections.splice(index, 1);
  }

  addResource() {
    this.resources.push('');
  }

  removeResource(index: number) {
    this.resources.splice(index, 1);
  }

  onSubmit() {
    if (this.moduleName) {
      const moduleId = this.moduleName.trim();    //Removing any illegal characters

      this.firestore.collection('Modules').doc(moduleId).set({
        name: this.moduleName,
        description: this.moduleDescription,
        id: this.id,
        sections: this.sections,
        resources: this.resources,
        videoUrl: this.videoUrl,
        imageUrl: this.imageUrl,
        createdAt: new Date()
      }).then(() => {
        this.dialogRef.close();
      }).catch((error) => {
        console.error('Error adding module: ', error);
      });
    }
  }

  onCancel() {
    this.dialogRef.close();
  }
}
