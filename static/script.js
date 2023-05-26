document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    // Verifica se a webcam está disponível
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.log('A webcam não está disponível.');
        return;
    }

    // Solicita acesso à webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            // Define a origem do vídeo como a webcam
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();

            // Processa o frame da webcam e exibe no canvas
            setInterval(() => {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            }, 0);
        })
        .catch((error) => {
            console.log('Erro ao acessar a webcam:', error);
        });
});
