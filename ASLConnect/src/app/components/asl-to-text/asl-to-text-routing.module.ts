import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AslToTextComponent } from './asl-to-text.component';

const routes: Routes = [
  {
    path: '',
    component: AslToTextComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class AslToTextRoutingModule { }
