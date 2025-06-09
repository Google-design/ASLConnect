import { Component } from '@angular/core';

export interface Resource {
  name: string;
  description?: string;
  link?: string;
  children?: Resource[];
}

@Component({
  selector: 'app-more-resources',
  templateUrl: './more-resources.component.html',
  styleUrls: ['./more-resources.component.scss']
})
export class MoreResourcesComponent {
  resources: Resource[] = [
    {
      name: 'Books on ASL',
      children: [
        { name: 'The American Sign Language Phrase Book by Lou Fant', description: 'A comprehensive phrasebook for beginner to advanced learners covering common ASL phrases and conversations.' },
        { name: 'Signing Naturally Student Workbook by Cheri Smith', description: 'A workbook with exercises and DVD content for interactive learning of ASL grammar and structure.' }
      ]
    },
    {
      name: 'ASL Video Content',
      children: [
        { name: 'ASL Connect YouTube Channel', link: 'https://www.youtube.com/ASLConnect', description: 'Video lessons organized by topic, covering everyday vocabulary and phrases.' },
        { name: 'ASL THAT YouTube Channel', link: 'https://www.youtube.com/ASLThat', description: 'Explains ASL grammar and linguistics with in-depth analysis from native ASL users.' }
      ]
    },
    {
      name: 'Online Courses',
      children: [
        { name: 'Gallaudet University’s ASL Connect Program', link: 'https://www.gallaudet.edu/ASLConnect', description: 'Offers comprehensive ASL courses ranging from basic to advanced. Students can earn certifications.' },
        { name: 'Start ASL Online Classes', link: 'https://www.startasl.com', description: 'Beginner to advanced courses for ASL learning, including specialized courses for interpreting.' }
      ]
    },
    {
      name: 'Mobile Apps',
      children: [
        { name: 'ASL Dictionary App (iOS & Android)', link: 'https://apps.apple.com/us/app/asl-dictionary/id579053524', description: 'Contains over 5,000 videos of ASL signs, including regional variations.' },
        { name: 'The ASL App', link: 'https://theaslapp.com', description: 'Created by Deaf ASL users, focusing on conversational ASL for everyday phrases and slang.' }
      ]
    },
    {
      name: 'Practice and Community Engagement',
      children: [
        { name: 'SignOn (Virtual Practice with Deaf Ambassadors)', link: 'https://signonconnect.com', description: 'Offers real-time practice sessions with Deaf ASL users for one-on-one conversations.' },
        { name: 'ASL Reddit Community', link: 'https://www.reddit.com/r/asl', description: 'A forum where learners and fluent ASL users share resources, questions, and practice opportunities.' }
      ]
    },
    {
      name: 'Deaf Culture and History',
      children: [
        { name: 'Documentaries', description: '"Through Deaf Eyes" - A documentary covering Deaf history and the evolution of Deaf culture in America.\n"The Silent Child" - A short film that highlights the importance of sign language in education for Deaf children.' },
        { name: 'Notable Figures', description: 'Laurent Clerc: Pioneering Deaf educator and co-founder of the first Deaf school in America.\nNyle DiMarco: Deaf activist and model promoting ASL and Deaf education worldwide.' }
      ]
    },
    {
      name: 'ASL Tools and Technology',
      children: [
        { name: 'Video Annotation Tools', description: 'ELAN (EUDICO Linguistic Annotator): Software for video analysis and annotation of ASL videos, useful for linguistics students.', link: 'https://archive.mpi.nl/tla/elan' },
        { name: 'Captioning Services', description: 'Rev.com: A transcription service that can add ASL-friendly captions to videos.', link: 'https://www.rev.com' },
        { name: 'Sign Language Dictionaries', description: 'ASL Pro Dictionary: An online dictionary with a searchable database of ASL words and phrases.', link: 'http://www.aslpro.com' }
      ]
    }
  ];
}
