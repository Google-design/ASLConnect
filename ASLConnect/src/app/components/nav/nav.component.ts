import { Component, inject, ViewChild } from '@angular/core';
import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';
import { Observable } from 'rxjs';
import { map, shareReplay } from 'rxjs/operators';
import { MatSidenav } from '@angular/material/sidenav';

@Component({
  selector: 'app-nav',
  templateUrl: './nav.component.html',
  styleUrl: './nav.component.scss'
})
export class NavComponent {
  @ViewChild('drawer') drawer!: MatSidenav;

  private breakpointObserver = inject(BreakpointObserver);

  isHandset$: Observable<boolean> = this.breakpointObserver.observe(Breakpoints.Handset)
    .pipe(
      map(result => result.matches),
      shareReplay()
    );

  navLinks: Array<any> = [
    {icon: 'dashboard', path: '', label: 'Home'},
    {icon: 'dashboard', path: '/modules', label: 'Modules'},
    {icon: 'table_rows', path: '/text-to-asl', label: 'Text-to-ASL'},
    {icon: 'today', path: '/asl-to-text', label: 'ASL-to-Text'},
    {icon: 'image', path: '/more-resources', label: 'More Resources'},
    {icon: 'image', path: '/contact-us', label: 'Contact Us'},
  ];
}
