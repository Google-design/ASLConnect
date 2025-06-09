export interface Module {
    name: string;
    description: string;
    sections: { heading: string, content: string}[];
    resources: string[];
    videoUrl?: string;
    imageUrl?: string;
    createdAt: Date;
    id: number;
}
