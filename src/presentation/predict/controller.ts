import { Request, Response } from "express";
import { AppController } from "../controller";
import { PredictService } from "../services";
import { CreatePredictDto } from "../../domain";

export class PredictController extends AppController {
    constructor(
        private readonly predictService: PredictService,
    ) { super(); }

    public realizarPredict = (req: Request, res: Response) => {
        const [error, createPredictdto] = CreatePredictDto.create(req.body);
        if(error || !createPredictdto) return res.status(400).json({ error });
        
        this.predictService.realizarPredict(createPredictdto) 
            .then(result => res.json(result))
            .catch(error => this.triggerError(error, res));
    }

}