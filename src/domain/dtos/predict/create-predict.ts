import { Validators } from "../../../config/validators";

export class CreatePredictDto {
    private constructor(
        public readonly glucosa: number,
        public readonly presion_arterial: number,
        public readonly grosor_de_piel: number,
        public readonly insulina: number,
        public readonly imc: number,
        public readonly diabetes_pedrigui_funcion: number,
    ) {}

    static create(data: {[key: string]: any}): [string?, CreatePredictDto?] {
        try {
            const validators = new Validators(data);

            validators.requiredKeys(
                'glucosa', 
                'presion_arterial', 
                'grosor_de_piel', 
                'insulina', 
                'imc',
                'diabetes_pedrigui_funcion'
            );

            validators.isNumber('glucosa');
            validators.isNumber('presion_arterial');
            validators.isNumber('grosor_de_piel');
            validators.isNumber('insulina');
            validators.isFloat('imc');
            validators.isNumber('diabetes_pedrigui_funcion');

            return [undefined, new CreatePredictDto(
                data['glucosa'],
                data['presion_arterial'],
                data['grosor_de_piel'],
                data['insulina'],
                data['imc'],
                data['diabetes_pedrigui_funcion'],
            )];
        } catch (error) {
            return [error as string];
        }
    }
}