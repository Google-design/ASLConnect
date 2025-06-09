import { Component, inject, model, OnInit } from '@angular/core';
import { Breakpoints, BreakpointObserver } from '@angular/cdk/layout';
import { map, switchMap } from 'rxjs/operators';
import { AngularFirestore } from '@angular/fire/compat/firestore';
import { Observable } from 'rxjs';
import { Module } from '../../services/module';

@Component({
  selector: 'app-modules',
  templateUrl: './modules.component.html',
  styleUrl: './modules.component.scss',
})

export class ModulesComponent implements OnInit{
  private breakpointObserver = inject(BreakpointObserver);
  modules$: Observable<Module[]>;   //Observable for modules

  constructor(private firestore: AngularFirestore) { }

  trackCard(index: number, card: any): number {
    return card.id; // Assuming each card has a unique 'id' property
  }

  ngOnInit(): void {
    // this.modules$ = this.firestore.collection<Module>('Modules').valueChanges();
    this.modules$ = this.firestore
      .collection<Module>('Modules', ref => ref.orderBy('id'))
      .valueChanges();
  }

  formatYTUrl(url?: string): string | undefined{
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
    return undefined;   //if no link
  }

  cards = this.breakpointObserver.observe(Breakpoints.Handset).pipe(
    switchMap(({ matches }) =>
      this.modules$.pipe(
        map(modules =>
          modules.map(module => ({
            title: module.name,
            content: module.description,
            sections: module.sections,
            resources: module.resources,
            videoUrl: this.formatYTUrl(module.videoUrl),
            imageUrl: module.imageUrl,
            id: module.id,
            cols: matches ? 1 : 2,
            rows: 1
          }))
        )
      )
    )
  );
}
