import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';
import { Component } from '@angular/core';
import { AngularFireStorage } from '@angular/fire/compat/storage';
import { forkJoin, Observable } from 'rxjs';

@Component({
  selector: 'app-text-to-asl',
  templateUrl: './text-to-asl.component.html',
  styleUrl: './text-to-asl.component.scss'
})
export class TextToAslComponent {
  cols: number;
  readonly showLetters: string = 'Show Letters';
  isChipSelected: boolean = false;    // Tracking the user's show letters chip
  aslImages$: Observable<string[]> | null = null; //For holding the imageURLS
  textInput: string = '';
  characters: string[] = []; // Array to hold characters corresponding to images
  constructor(private storage: AngularFireStorage, private breakpointOserver: BreakpointObserver) {
    breakpointOserver.observe([Breakpoints.Small, Breakpoints.HandsetPortrait]).subscribe(res => {
      this.cols = res.matches ? 1 : 2;
    })
  }

  onSubmit() {
    const observables: Observable<string>[] = [];
    const text = this.textInput.toUpperCase();
    this.characters = []; // Resetting the characters array

    for(let char of text){
      if (/[A-Z0-9]/.test(char)) { // Only images for A-Z and 0-9
        this.characters.push(char); // Add character to array
        const path = `english-asl-images/${char}.png`;
        const imageUrl = this.storage.ref(path).getDownloadURL(); // Fetching image URL
        observables.push(imageUrl); // Store observable in array
      }
    }
    if (observables.length > 0) {
      this.aslImages$ = forkJoin(observables); // Assign the forkJoin result to aslImages$
    } else {
      this.aslImages$ = null; // Reset if no valid input
    }
  }

  chipChange() {
    this.isChipSelected = !this.isChipSelected;
  }
}
