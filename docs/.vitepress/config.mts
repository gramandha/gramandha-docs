import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: 'gramandha-docs',
  description: 'Modul pelatihan Gramandha',
  base: '/gramandha-docs/',
  cleanUrls: true,
  appearance: 'default',
  vite: {
    optimizeDeps: {
      include: ['fastdom', 'fastdom/extensions/fastdom-promised.js']
    }
  },
  themeConfig: {
    nav: [
      { text: 'Beranda', link: '/' },
      { text: 'ROS', link: '/ros/00-pendahuluan' },
      { text: 'YOLO', link: '/yolo/00-pendahuluan' },
      { text: 'Bahasa C', link: '/bahasa-c/Modul-0-Algoritma-Flowchart-Pseudocode' }
    ],
    sidebar: {
      '/ros/': [
        {
          text: 'Modul ROS 2 Jazzy',
          collapsed: false,
          items: [
            { text: 'Pendahuluan', link: '/ros/00-pendahuluan' },
            { text: 'Lesson 1 - Instalasi ROS 2', link: '/ros/01-lesson-1-instalasi-ros2' },
            { text: 'Lesson 2 - Workspace & Package', link: '/ros/02-lesson-2-workspace-package' },
            { text: 'Lesson 3 - Node Python', link: '/ros/03-lesson-3-node-python' },
            { text: 'Lesson 4 - Topic & Message', link: '/ros/04-lesson-4-topic-message' },
            { text: 'Lesson 5 - Komunikasi Antar Node', link: '/ros/05-lesson-5-komunikasi-antar-node' },
            { text: 'Lesson 6 - Integrasi Kamera', link: '/ros/06-lesson-6-integrasi-kamera' },
            { text: 'Lesson 7 - Integrasi YOLO', link: '/ros/07-lesson-7-integrasi-yolo' },
            { text: 'Lesson 8 - Evaluasi Referensi', link: '/ros/08-evaluasi-referensi' },
            
          ]
        }
      ],
      '/yolo/': [
        {
          text: 'Modul Pembelajaran YOLO',
          collapsed: false,
          items: [
            { text: 'Pendahuluan', link: '/yolo/00-pendahuluan' },
            { text: 'Lesson 1 - Prasyarat Dasar', link: '/yolo/01-lesson-1-prasyarat-dasar' },
            { text: 'Lesson 2 - Konsep Dasar Object Detection', link: '/yolo/02-lesson-2-konsep-object-detection' },
            { text: 'Lesson 3 - Arsitektur Evolusi YOLO', link: '/yolo/03-lesson-3-arsitektur-evolusi-yolo' },
            { text: 'Lesson 4 - Praktik dan Inferensi Model Pretrained', link: '/yolo/04-lesson-4-inferensi-pretrained' },
            { text: 'Lesson 5 - Pelatihan Dataset Kustom', link: '/yolo/05-lesson-5-pelatihan-dataset-kustom' },
            { text: 'Lesson 6 - Optimasi dan Deployment', link: '/yolo/06-lesson-6-optimasi-deployment' },
            { text: 'Evaluasi Akhir dan Referensi', link: '/yolo/07-evaluasi-referensi' }
          ]
        }
      ],
      '/bahasa-c/': [
        {
          text: 'Modul Bahasa C',
          collapsed: false,
          items: [
            { text: 'Pendahuluan', link: '/bahasa-c/Modul-0-Algoritma-Flowchart-Pseudocode' },
            { text: 'Modul 1 - Fondasi Bahasa C', link: '/bahasa-c/Modul-1-Fondasi-Bahasa-C' },
            { text: 'Modul 2 - Kontrol Alur Logika', link: '/bahasa-c/Modul-2-Kontrol-Alur-Logika' },
            { text: 'Modul 3 - Fungsi dan Cakupan Variabel', link: '/bahasa-c/Modul-3-Fungsi-dan-Cakupan-Variabel' },
            { text: 'Modul 4 - Struktur Data Dasar', link: '/bahasa-c/Modul-4-Struktur-Data-Dasar' },
            { text: 'Modul 5 - Pointer dan Memori Dinamis', link: '/bahasa-c/Modul-5-Pointer-dan-Memori-Dinamis' },
            { text: 'Modul 6 - Contoh Implementasi Algoritma Dasar', link: '/bahasa-c/Modul-6-Contoh-Implentasi-Algoritma-Dasar' },
            { text: 'Modul 7 - Operasi Berkas', link: '/bahasa-c/Modul-7-Operasi-Berkas' },
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/' }
    ]
  }
}))
