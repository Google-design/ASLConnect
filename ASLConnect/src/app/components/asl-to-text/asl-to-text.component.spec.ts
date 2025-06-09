import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AslToTextComponent } from './asl-to-text.component';

describe('AslToTextComponent', () => {
  let component: AslToTextComponent;
  let fixture: ComponentFixture<AslToTextComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [AslToTextComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AslToTextComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
