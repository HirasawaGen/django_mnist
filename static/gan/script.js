setInterval(() => {
    fetch('/gan-process/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    }).then(response => {
        return response.json();
    }).then(data => {
        for (let i = 0; i < 10; i++) {
            document.getElementById(`image_${i}`).src = data.fake_images[i];
        }
        console.log(data.pos_probs);
    }).catch(error => {
        console.error(error);
    });
}, 1000);