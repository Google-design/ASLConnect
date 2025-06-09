import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { CameraViewComponent } from './components/camera-view/camera-view.component'; // Adjust the path as necessary


const routes: Routes = [
  {
    path: '',
    loadChildren: () => import('./components/nav/nav.module').then(m => m.NavModule)
  },
  // {
  //   path: 'login',
  //   loadChildren: () => import('./components/login/login.module').then(m => m.LoginModule)
  // },
  {
    path: 'camera',
    component: CameraViewComponent
  },
  {
    path: '**',
    loadChildren: () => import('./components/pagenotfound/pagenotfound.module').then(m => m.PagenotfoundModule)
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
