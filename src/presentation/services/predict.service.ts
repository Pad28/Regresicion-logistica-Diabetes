import { CreatePredictDto } from "../../domain";
import { execSync } from "child_process";
import fs from 'fs';

export class PredictService {
    constructor() {}

    public async realizarPredict (createPredictDto: CreatePredictDto) {
        fs.writeFileSync('./data.json', JSON.stringify({
            "Glucosa": createPredictDto.glucosa,
            "Presion arterial": createPredictDto.presion_arterial,
            "Grosor de la piel": createPredictDto.grosor_de_piel,
            "Insulina": createPredictDto.insulina,
            "IMC": createPredictDto.imc,
            "DiabetesPedigríFunción": createPredictDto.diabetes_pedrigui_funcion,
        }));
        execSync('python3 model.py');
        return JSON.parse(
            fs.readFileSync('./prediccion.json', { encoding: 'utf8' })
        );
    }
}