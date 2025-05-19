var w = 800;
var h = 400;
var jugador, fondo, bala, nave;
var balaD = false;
var salto, menu;
var velocidadBala, despBala;
var estatusAire, estatuSuelo;

var nnNetwork, nnEntrenamiento, nnSalida, datosEntrenamiento = [];
var modoAuto = false, eCompleto = false;
var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload, create, update, render });

function preload() {
    juego.load.image('fondo', 'assets/game/fondo.jpg');
    juego.load.spritesheet('mono', 'assets/sprites/altair.png', 32, 48);
    juego.load.image('nave', 'assets/game/ufo.png');
    juego.load.image('bala', 'assets/sprites/purple_ball.png');
    juego.load.image('menu', 'assets/game/menu.png');
}

function create() {
    juego.physics.startSystem(Phaser.Physics.ARCADE);
    juego.physics.arcade.gravity.y = 800;
    juego.time.desiredFps = 30;

    fondo = juego.add.tileSprite(0, 0, w, h, 'fondo');
    nave = juego.add.sprite(w - 100, h - 70, 'nave');
    bala = juego.add.sprite(w - 100, h, 'bala');
    jugador = juego.add.sprite(50, h, 'mono');

    juego.physics.enable(jugador);
    jugador.body.collideWorldBounds = true;
    jugador.animations.add('corre', [8, 9, 10, 11]);
    jugador.animations.play('corre', 10, true);

    juego.physics.enable(bala);
    bala.body.collideWorldBounds = true;

    pausaL = juego.add.text(w - 100, 20, 'Pausa', { font: '20px Arial', fill: '#fff' });
    pausaL.inputEnabled = true;
    pausaL.events.onInputUp.add(pausa, this);
    juego.input.onDown.add(mPausa, this);

    salto = juego.input.keyboard.addKey(Phaser.Keyboard.SPACEBAR);

    // Red neuronal simple: 2 entradas, 6 neuronas ocultas, 1 salida
    nnNetwork = new synaptic.Architect.Perceptron(2, 10, 10, 1);
    nnEntrenamiento = new synaptic.Trainer(nnNetwork);
}

function normalizar(x) {
    // Normaliza valor entre 0 y 1
    return (x + 800) / 1600;
}

function enRedNeural() {
    console.log("Entrenando con " + datosEntrenamiento.length + " ejemplos...");
    nnEntrenamiento.train(datosEntrenamiento, {
        rate: 0.1,
        iterations: 20000,
        shuffle: true
    });
    console.log("Entrenamiento completo.");
}

function datosDeEntrenamiento(param_entrada) {
    nnSalida = nnNetwork.activate(param_entrada);
    var resultado = nnSalida[0]; // 0 a 1
    console.log("Entrada:", param_entrada, "| Salida:", resultado);
    console.log(`Desp: ${param_entrada[0]}, Vel: ${param_entrada[1]} => Salida: ${resultado.toFixed(3)}`);
    return resultado > 0.5; // Saltar si alta probabilidad
}

function pausa() {
    juego.paused = true;
    menu = juego.add.sprite(w / 2, h / 2, 'menu');
    menu.anchor.setTo(0.5, 0.5);
}

function mPausa(event) {
    if (juego.paused) {
        var menu_x1 = w / 2 - 135, menu_x2 = w / 2 + 135;
        var menu_y1 = h / 2 - 90, menu_y2 = h / 2 + 90;

        var mouse_x = event.x, mouse_y = event.y;

        if (mouse_x > menu_x1 && mouse_x < menu_x2 && mouse_y > menu_y1 && mouse_y < menu_y2) {
            if (mouse_y <= menu_y1 + 90) {
                eCompleto = false;
                datosEntrenamiento = [];
                modoAuto = false;
            } else {
                if (!eCompleto) {
                    enRedNeural();
                    eCompleto = true;
                }
                modoAuto = true;
            }
            menu.destroy();
            resetVariables();
            juego.paused = false;
        }
    }
}

function resetVariables() {
    jugador.body.velocity.x = 0;
    jugador.body.velocity.y = 0;
    bala.body.velocity.x = 0;
    bala.position.x = w - 100;
    jugador.position.x = 50;
    balaD = false;
}

function saltar() {
    jugador.body.velocity.y = -270;
}

function update() {
    fondo.tilePosition.x -= 1;

    juego.physics.arcade.collide(bala, jugador, colisionH, null, this);

    estatuSuelo = jugador.body.onFloor() ? 1 : 0;
    estatusAire = jugador.body.onFloor() ? 0 : 1;

    despBala = Math.floor(jugador.position.x - bala.position.x);

    // Salto manual
    if (!modoAuto && salto.isDown && jugador.body.onFloor()) {
        saltar();
        // Entrenamiento positivo (saltó a tiempo)
        if (despBala < -30) {
            datosEntrenamiento.push({
                input: [normalizar(despBala), normalizar(velocidadBala)],
                output: [1] // debe saltar
            });
        }
    }

    // IA decide
    if (modoAuto && bala.position.x > 0 && jugador.body.onFloor()) {
        if (datosDeEntrenamiento([normalizar(despBala), normalizar(velocidadBala)])) {
            saltar();
        }
    }

    // Disparo inicial
    if (!balaD) {
        disparo();
    }

    // Si la bala ya salió de pantalla, reiniciar
    if (bala.position.x <= 0) {
        resetVariables();
    }

    // Entrenamiento negativo (no saltó, pero estaba cerca)
    if (!modoAuto && bala.position.x > 0 && despBala < -30 && jugador.body.onFloor()) {
        datosEntrenamiento.push({
            input: [normalizar(despBala), normalizar(velocidadBala)],
            output: [0] // no debe saltar
        });
    }
}

function disparo() {
    velocidadBala = -1 * velocidadRandom(300, 800);
    bala.body.velocity.y = 0;
    bala.body.velocity.x = velocidadBala;
    balaD = true;
}

function colisionH() {
    if (!modoAuto) {
        // Entrenamiento negativo (falló al saltar)
        datosEntrenamiento.push({
            input: [normalizar(despBala), normalizar(velocidadBala)],
            output: [0]
        });
    }
    pausa();
}


function velocidadRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function render() {
    // Muestra debug si lo deseas
}
