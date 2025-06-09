import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatChipsModule } from '@angular/material/chips';
import { AslToTextRoutingModule } from './asl-to-text-routing.module';

@NgModule({
  declarations: [
  ],
  imports: [
    CommonModule,
    AslToTextRoutingModule,
    MatChipsModule
  ]
})
export class AslToTextModule { }