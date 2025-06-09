import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { TextToAslComponent } from './text-to-asl.component';

const routes: Routes = [
  {
    path: '',
    component: TextToAslComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class TextToAslRoutingModule { }
