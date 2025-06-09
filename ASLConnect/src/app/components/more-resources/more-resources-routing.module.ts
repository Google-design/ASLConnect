import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { MoreResourcesComponent } from './more-resources.component';

const routes: Routes = [
  {
    path: '',
    component: MoreResourcesComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class MoreResourcesRoutingModule { }
