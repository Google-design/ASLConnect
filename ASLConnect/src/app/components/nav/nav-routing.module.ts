import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { NavComponent } from './nav.component';
import { redirectLoggedInTo, redirectUnauthorizedTo, canActivate} from '@angular/fire/compat/auth-guard';

const redirectUnauthorizedToLogin = () => redirectUnauthorizedTo(['login']);
const redirectLoggedInToHome = () => redirectLoggedInTo(['admin']);

const routes: Routes = [
  {
    path: '',
    component: NavComponent,
    children: [
      {
        path: '',
        loadChildren: () => import('../default/default.module').then(m => m.DefaultModule),
      },
      {
        path: 'modules',
        loadChildren: () => import('../modules/modules.module').then(m => m.ModulesModule),
        //component: ModulesComponent
      },
      {
        path: 'text-to-asl',
        loadChildren: () => import('../text-to-asl/text-to-asl.module').then(m => m.TextToAslModule)

      },
      {
        path: 'asl-to-text',
        loadChildren: () => import('../asl-to-text/asl-to-text.module').then(m => m.AslToTextModule)

      },
      {
        path: 'more-resources',
        loadChildren: () => import('../more-resources/more-resources.module').then(m => m.MoreResourcesModule)
      },
      {
        path: 'contact-us',
        loadChildren: () => import('../contact-us/contact-us.module').then(m => m.ContactUsModule),
      },
      {
        path: 'admin',
        loadChildren: () => import('../admin/admin.module').then(m => m.AdminModule),
        ...canActivate(redirectUnauthorizedToLogin)
      },
      {
        path: 'login',
        loadChildren: () => import('../login/login.module').then(m => m.LoginModule),
        ...canActivate(redirectLoggedInToHome)
      },
    ]
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class NavRoutingModule { }
