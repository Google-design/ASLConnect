import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TextToAslComponent } from './text-to-asl.component';

describe('TextToAslComponent', () => {
  let component: TextToAslComponent;
  let fixture: ComponentFixture<TextToAslComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [TextToAslComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TextToAslComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
